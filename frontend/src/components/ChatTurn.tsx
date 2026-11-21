import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Answer, EvidencePayload, Stage } from "../api";
import { Abstained, AnswerText, EmptyResult, ErrorState, StageProgress } from "./AnswerPanel";
import { EvidencePanel } from "./EvidencePanel";

export interface TurnData {
  question: string;
  text: string; // streamed or final answer text
  answer: Answer | null;
  evidence: EvidencePayload | null;
  status: "retrieving" | "reranking" | "generating" | "answered" | "abstained" | "empty" | "chitchat" | "error";
  errorMessage?: string;
}

const BUSY = new Set(["retrieving", "reranking", "generating"]);

export function ChatTurnView({ turn, latest }: { turn: TurnData; latest: boolean }) {
  const [activeMarker, setActiveMarker] = useState<number | null>(null);
  const busy = BUSY.has(turn.status);
  const showEvidence =
    turn.evidence && turn.evidence.items.length > 0 &&
    turn.status !== "error" && turn.status !== "chitchat";

  return (
    <article className="border-b border-neutral-100 py-6 last:border-b-0">
      <h2 className="text-sm font-semibold text-neutral-900">{turn.question}</h2>

      <div className={"mt-3 grid gap-6 " + (showEvidence ? "md:grid-cols-[minmax(0,1fr)_300px]" : "")}>
        <section>
          {busy && (
            <div className="mb-3">
              <StageProgress stage={turn.status as Stage} />
            </div>
          )}
          {turn.status === "error" && <ErrorState message={turn.errorMessage ?? "Unknown error"} />}
          {turn.status === "abstained" && <Abstained />}
          {turn.status === "empty" && <EmptyResult />}
          {turn.status === "chitchat" && (
            <div className="text-[15px] leading-7 text-neutral-800 space-y-2">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.text}</ReactMarkdown>
            </div>
          )}
          {(turn.status === "generating" || turn.status === "answered") && turn.text && (
            <AnswerText
              text={turn.text}
              streaming={turn.status === "generating"}
              activeMarker={activeMarker}
              onMarkerHover={setActiveMarker}
              onMarkerClick={setActiveMarker}
            />
          )}
        </section>

        {showEvidence && (
          <CollapsibleEvidence
            turn={turn}
            latest={latest}
            activeMarker={activeMarker}
            onCardHover={setActiveMarker}
          />
        )}
      </div>
    </article>
  );
}

function CollapsibleEvidence({
  turn, latest, activeMarker, onCardHover,
}: {
  turn: TurnData;
  latest: boolean;
  activeMarker: number | null;
  onCardHover: (n: number | null) => void;
}) {
  const [expanded, setExpanded] = useState(latest);
  if (!turn.evidence) return null;
  if (!expanded) {
    return (
      <button
        onClick={() => setExpanded(true)}
        className="self-start rounded-lg border border-neutral-200 px-3 py-1.5 text-xs text-neutral-500 hover:bg-neutral-100"
        aria-expanded={false}
      >
        Show evidence ({turn.evidence.items.length})
      </button>
    );
  }
  return (
    <EvidencePanel
      evidence={turn.evidence}
      citedIds={turn.answer?.used_chunk_ids ?? []}
      activeMarker={activeMarker}
      onCardHover={onCardHover}
    />
  );
}
