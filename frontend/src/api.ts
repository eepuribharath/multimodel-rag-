const API_BASE =
  typeof window !== "undefined" && window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : "http://localhost:8000";

export interface Citation {
  chunk_id: string;
  source_id: string;
  locator: string;
  modality: string;
}

export interface Answer {
  text: string;
  citations: Citation[];
  used_chunk_ids: string[];
  abstained: boolean;
  kind: "grounded" | "abstained" | "chitchat";
}

export interface ChatSummary {
  id: number;
  title: string;
  updated_at: string;
  turns: number;
}

export interface ChatTurn {
  id: number;
  question: string;
  answer: Answer;
  evidence: EvidencePayload | null;
  created_at: string;
}

export interface EvidenceItem {
  chunk_id: string;
  source_id: string;
  modality: string;
  locator: string;
  snippet: string;
  media_url: string | null;
  start_ms: number | null;
  rrf_score: number;
  rerank_score: number | null;
}

export interface CandidateRow {
  chunk_id: string;
  source_id: string;
  modality: string;
  rrf_score: number;
}

export interface EvidencePayload {
  items: EvidenceItem[];
  pre_rerank: CandidateRow[];
}

export type Stage = "retrieving" | "reranking" | "generating";

export interface StreamHandlers {
  onChat?: (chatId: number) => void; // server created a chat for this query
  onStage: (stage: Stage) => void;
  onEvidence: (payload: EvidencePayload) => void;
  onDelta: (text: string) => void;
  onAnswer: (answer: Answer) => void;
  onError: (message: string) => void;
}

export function mediaUrl(path: string): string {
  return `${API_BASE}${path}`;
}

/** POST /query/stream and dispatch SSE events. Uses fetch + ReadableStream
 * (not EventSource) because the endpoint takes a POST body. */
export async function streamQuery(
  question: string,
  chatId: number | null,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, chat_id: chatId }),
      signal,
    });
  } catch (err) {
    if ((err as Error).name !== "AbortError")
      handlers.onError("Cannot reach the backend — is the API server running?");
    return;
  }
  if (!response.ok || !response.body) {
    handlers.onError(`Backend error (${response.status})`);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (frame: string) => {
    let event = "";
    const data: string[] = [];
    for (const line of frame.split("\n")) {
      if (line.startsWith("event: ")) event = line.slice(7);
      else if (line.startsWith("data: ")) data.push(line.slice(6));
    }
    const payload = data.join("\n");
    switch (event) {
      case "chat":
        handlers.onChat?.(JSON.parse(payload).chat_id);
        break;
      case "stage":
        handlers.onStage(payload as Stage);
        break;
      case "evidence":
        handlers.onEvidence(JSON.parse(payload));
        break;
      case "delta":
        handlers.onDelta(payload);
        break;
      case "answer":
        handlers.onAnswer(JSON.parse(payload));
        break;
      case "error":
        handlers.onError(JSON.parse(payload).message ?? "Unknown error");
        break;
    }
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      dispatch(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");
    }
  }
}

export async function listChats(): Promise<ChatSummary[]> {
  const response = await fetch(`${API_BASE}/chats`);
  if (!response.ok) throw new Error(`Failed to load chats (${response.status})`);
  return response.json();
}

export async function createChat(): Promise<ChatSummary> {
  const response = await fetch(`${API_BASE}/chats`, { method: "POST" });
  if (!response.ok) throw new Error(`Failed to create chat (${response.status})`);
  return response.json();
}

export async function getChatTurns(chatId: number): Promise<ChatTurn[]> {
  const response = await fetch(`${API_BASE}/chats/${chatId}`);
  if (!response.ok) throw new Error(`Failed to load chat (${response.status})`);
  return response.json();
}

export async function deleteChat(chatId: number): Promise<void> {
  const response = await fetch(`${API_BASE}/chats/${chatId}`, { method: "DELETE" });
  if (!response.ok) throw new Error(`Failed to delete chat (${response.status})`);
}

export interface IngestResult {
  source_id: string;
  chunks: number;
}

export async function ingestFile(file: File): Promise<IngestResult> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}/ingest`, { method: "POST", body: form });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Ingest failed (${response.status})`);
  }
  return response.json();
}
