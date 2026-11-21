"""Eval harness (spec 06): retrieval quality, reranker lift, citation
integrity, and abstention on a golden set, gated against baselines.json.

Usage (from repo root, inside the venv):
    python eval/run_eval.py                     # run + compare to baselines
    python eval/run_eval.py --update-baselines  # write observed - margin as new floors
    python eval/run_eval.py --full              # also run Ragas answer metrics

Exits non-zero if any metric falls below its committed baseline floor — wire
this into CI. The corpus is indexed into an in-memory collection per run, so
results never depend on leftover local state.
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel
from qdrant_client import QdrantClient

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from metrics import ndcg_at_k, recall_at_k  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.embedding import get_embedder  # noqa: E402
from app.generation import get_generator  # noqa: E402
from app.ingestion import ingest  # noqa: E402
from app.rerank import get_reranker  # noqa: E402
from app.retrieval import VectorStore  # noqa: E402

BASELINES_PATH = EVAL_DIR / "baselines.json"
REPORT_PATH = EVAL_DIR / "report.json"
BASELINE_MARGIN = 0.05  # floors sit this far under observed values


class EvalReport(BaseModel):
    config: dict
    n_questions: int
    n_out_of_corpus: int
    recall_at_100: float
    ndcg_at_10_pre_rerank: float
    ndcg_at_6_pre_rerank: float
    ndcg_at_6_post_rerank: float
    reranker_lift: float  # post - pre at k=6
    citation_valid_rate: float  # answerable questions with ≥1 valid citation
    abstention_rate: float  # out-of-corpus questions correctly abstained
    ragas: dict | None = None
    per_question: list[dict]


def build_report(dataset_path: str, full: bool = False) -> EvalReport:
    settings = get_settings()
    dataset = json.loads(Path(dataset_path).read_text())
    corpus_dir = Path(dataset_path).parent / "corpus"

    embedder = get_embedder(settings)
    reranker = get_reranker(settings)
    generator = get_generator(settings)
    store = VectorStore(
        client=QdrantClient(":memory:"),
        collection="eval",
        dim=settings.embed_dim,
        rrf_k=settings.rrf_k,
    )

    for doc in sorted(corpus_dir.glob("*.md")):
        chunks = ingest(str(doc))
        store.upsert(chunks, embedder.embed_chunks(chunks))

    from app.schemas import Query

    per_question: list[dict] = []
    recalls, ndcg10, ndcg6_pre, ndcg6_post, citation_ok = [], [], [], [], []
    answers = []  # (question, ground_truth, contexts, answer_text) for ragas

    for entry in dataset["items"]:
        question, expected = entry["question"], entry["expected_source_ids"]
        query_vector = embedder.embed_query(Query(text=question))
        candidates = store.hybrid_search(
            query_vector, question, top_k=settings.retrieve_top_k
        )
        sources_pre = [item.chunk.source_id for item in candidates]
        reranked = reranker.rerank(question, candidates, top_k=settings.rerank_top_k)
        sources_post = [item.chunk.source_id for item in reranked]

        recalls.append(recall_at_k(sources_pre, expected, 100))
        ndcg10.append(ndcg_at_k(sources_pre, expected, 10))
        ndcg6_pre.append(ndcg_at_k(sources_pre, expected, 6))
        ndcg6_post.append(ndcg_at_k(sources_post, expected, 6))

        answer = generator.generate(question, reranked)
        valid = not answer.abstained and any(
            c.source_id in expected for c in answer.citations
        )
        citation_ok.append(1.0 if valid else 0.0)
        answers.append(
            {
                "question": question,
                "ground_truth": entry["ground_truth"],
                "contexts": [i.chunk.content_ref for i in reranked],
                "answer": answer.text,
            }
        )
        per_question.append(
            {
                "question": question,
                "kind": entry.get("kind", ""),
                "recall@100": recalls[-1],
                "ndcg@6_pre": ndcg6_pre[-1],
                "ndcg@6_post": ndcg6_post[-1],
                "citation_valid": bool(valid),
                "abstained": answer.abstained,
            }
        )

    abstained = []
    for entry in dataset["out_of_corpus"]:
        question = entry["question"]
        query_vector = embedder.embed_query(Query(text=question))
        candidates = store.hybrid_search(
            query_vector, question, top_k=settings.retrieve_top_k
        )
        reranked = reranker.rerank(question, candidates, top_k=settings.rerank_top_k)
        answer = generator.generate(question, reranked)
        abstained.append(1.0 if answer.abstained else 0.0)
        per_question.append(
            {"question": question, "kind": "out_of_corpus", "abstained": answer.abstained}
        )

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    report = EvalReport(
        config={
            "embed_model": settings.embed_model,
            "embed_dim": settings.embed_dim,
            "gen_model": settings.gen_model,
            "reranker": settings.reranker,
            "rerank_model": settings.rerank_model,
            "retrieve_top_k": settings.retrieve_top_k,
            "rerank_top_k": settings.rerank_top_k,
        },
        n_questions=len(dataset["items"]),
        n_out_of_corpus=len(dataset["out_of_corpus"]),
        recall_at_100=mean(recalls),
        ndcg_at_10_pre_rerank=mean(ndcg10),
        ndcg_at_6_pre_rerank=mean(ndcg6_pre),
        ndcg_at_6_post_rerank=mean(ndcg6_post),
        reranker_lift=mean(ndcg6_post) - mean(ndcg6_pre),
        citation_valid_rate=mean(citation_ok),
        abstention_rate=mean(abstained),
        ragas=run_ragas(answers) if full else None,
        per_question=per_question,
    )
    return report


def run_ragas(answers: list[dict]) -> dict | None:
    """Ragas answer metrics with Gemini as judge. Optional: needs the ragas +
    langchain-google-genai extras and burns judge quota (~4 calls/question)."""
    try:
        from datasets import Dataset
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig
    except ImportError as exc:
        print(f"[ragas] skipped — extras not installed ({exc.name}). "
              "Install: pip install ragas langchain-google-genai")
        return None

    settings = get_settings()
    judge = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=settings.gen_model, google_api_key=settings.gemini_api_key)
    )
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model=settings.embed_model, google_api_key=settings.gemini_api_key
        )
    )
    result = evaluate(
        Dataset.from_list(answers),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=1),  # free-tier RPM is tight
    )
    return {k: float(v) for k, v in result.scores.to_pandas().mean().items()}


GATED_METRICS = (
    "recall_at_100",
    "ndcg_at_10_pre_rerank",
    "ndcg_at_6_post_rerank",
    "citation_valid_rate",
    "abstention_rate",
)


def compare_to_baselines(report: EvalReport) -> list[str]:
    if not BASELINES_PATH.exists():
        print("[baselines] none committed yet — run with --update-baselines to set them")
        return []
    baselines = json.loads(BASELINES_PATH.read_text())
    if baselines.get("config") != report.config:
        print("[baselines] WARNING: config changed since baselines were set — "
              "numbers are not comparable; re-run --update-baselines deliberately")
    failures = []
    for metric, floor in baselines["floors"].items():
        observed = getattr(report, metric)
        if observed < floor:
            failures.append(f"{metric}: {observed:.3f} < floor {floor:.3f}")
    return failures


def update_baselines(report: EvalReport) -> None:
    floors = {
        m: round(max(0.0, getattr(report, m) - BASELINE_MARGIN), 3)
        for m in GATED_METRICS
    }
    BASELINES_PATH.write_text(
        json.dumps({"config": report.config, "floors": floors}, indent=2) + "\n"
    )
    print(f"[baselines] written to {BASELINES_PATH}")


def run_eval(dataset_path: str, full: bool = False, update: bool = False) -> int:
    report = build_report(dataset_path, full=full)

    print(f"\n=== eval: {report.n_questions} questions, "
          f"{report.n_out_of_corpus} out-of-corpus ===")
    for metric in ("recall_at_100", "ndcg_at_10_pre_rerank", "ndcg_at_6_pre_rerank",
                   "ndcg_at_6_post_rerank", "reranker_lift",
                   "citation_valid_rate", "abstention_rate"):
        print(f"  {metric:26s} {getattr(report, metric):+.3f}")
    if report.ragas:
        for k, v in report.ragas.items():
            print(f"  ragas.{k:20s} {v:+.3f}")

    REPORT_PATH.write_text(report.model_dump_json(indent=2) + "\n")
    print(f"[report] written to {REPORT_PATH}")

    if update:
        update_baselines(report)
        return 0
    failures = compare_to_baselines(report)
    for failure in failures:
        print(f"[FAIL] {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(EVAL_DIR / "datasets" / "golden.json"))
    parser.add_argument("--full", action="store_true", help="include Ragas answer metrics")
    parser.add_argument("--update-baselines", action="store_true")
    args = parser.parse_args()
    sys.exit(run_eval(args.dataset, full=args.full, update=args.update_baselines))
