import { useCallback, useEffect, useRef, useState } from "react";
import type { ChatSummary, EvidencePayload } from "./api";
import {
  createChat,
  deleteChat,
  getChatTurns,
  listChats,
  streamQuery,
} from "./api";
import { QueryBar } from "./components/QueryBar";
import { ChatTurnView, type TurnData } from "./components/ChatTurn";
import { Sidebar } from "./components/Sidebar";

export default function App() {
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [activeChatId, setActiveChatId] = useState<number | null>(null);
  const [turns, setTurns] = useState<TurnData[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const refreshChats = useCallback(async () => {
    try {
      setChats(await listChats());
    } catch {
      /* backend down — the query path surfaces the error state */
    }
  }, []);

  useEffect(() => {
    void refreshChats();
  }, [refreshChats]);

  const openChat = useCallback(async (chatId: number) => {
    abortRef.current?.abort();
    setActiveChatId(chatId);
    setSidebarOpen(false);
    try {
      const loaded = await getChatTurns(chatId);
      setTurns(
        loaded.map((turn) => ({
          question: turn.question,
          text: turn.answer.text,
          answer: turn.answer,
          evidence: turn.evidence,
          status:
            turn.answer.kind === "chitchat" ? "chitchat"
            : turn.answer.kind === "abstained" ? "abstained"
            : "answered",
        })),
      );
    } catch {
      setTurns([]);
    }
  }, []);

  const newChat = useCallback(async () => {
    abortRef.current?.abort();
    try {
      const chat = await createChat();
      setActiveChatId(chat.id);
      setTurns([]);
      void refreshChats();
    } catch {
      // fall back to "no chat yet" — first query will create one
      setActiveChatId(null);
      setTurns([]);
    }
    setSidebarOpen(false);
  }, [refreshChats]);

  const removeChat = useCallback(
    async (chatId: number) => {
      if (!window.confirm("Delete this chat and its history?")) return;
      await deleteChat(chatId).catch(() => {});
      if (chatId === activeChatId) {
        setActiveChatId(null);
        setTurns([]);
      }
      void refreshChats();
    },
    [activeChatId, refreshChats],
  );

  const patchLastTurn = useCallback((patch: Partial<TurnData>) => {
    setTurns((prev) => {
      if (prev.length === 0) return prev;
      const next = prev.slice();
      next[next.length - 1] = { ...next[next.length - 1], ...patch };
      return next;
    });
  }, []);

  const ask = useCallback(
    (question: string) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setTurns((prev) => [
        ...prev,
        { question, text: "", answer: null, evidence: null, status: "retrieving" },
      ]);

      let evidence: EvidencePayload | null = null;
      let sawDelta = false;
      void streamQuery(
        question,
        activeChatId,
        {
          onChat: (chatId) => {
            setActiveChatId(chatId);
            void refreshChats();
          },
          onStage: (stage) => patchLastTurn({ status: stage }),
          onEvidence: (payload) => {
            evidence = payload;
            patchLastTurn({ evidence: payload });
          },
          onDelta: (delta) => {
            sawDelta = true;
            setTurns((prev) => {
              const next = prev.slice();
              const last = next[next.length - 1];
              next[next.length - 1] = { ...last, text: last.text + delta };
              return next;
            });
          },
          onAnswer: (answer) => {
            const status =
              answer.kind === "chitchat" ? "chitchat"
              : answer.kind === "abstained"
                ? (evidence && evidence.items.length === 0) || (!evidence && !sawDelta)
                  ? "empty"
                  : "abstained"
                : "answered";
            patchLastTurn({ answer, text: answer.text, status });
            void refreshChats(); // titles/timestamps moved
          },
          onError: (message) =>
            patchLastTurn({ status: "error", errorMessage: message }),
        },
        controller.signal,
      );
    },
    [activeChatId, patchLastTurn, refreshChats],
  );

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length, turns[turns.length - 1]?.text]);

  const busy = ["retrieving", "reranking", "generating"].includes(
    turns[turns.length - 1]?.status ?? "",
  );
  const hasThread = turns.length > 0;

  return (
    <div className="flex h-screen bg-neutral-50 text-neutral-900 antialiased">
      {/* screen-reader announcements */}
      <div className="sr-only" aria-live="polite" role="status">
        {busy ? `Stage: ${turns[turns.length - 1].status}` : ""}
      </div>

      <Sidebar
        chats={chats}
        activeId={activeChatId}
        open={sidebarOpen}
        onSelect={(id) => void openChat(id)}
        onNew={() => void newChat()}
        onDelete={(id) => void removeChat(id)}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        {/* mobile top bar */}
        <div className="flex items-center gap-2 border-b border-neutral-200 bg-white px-3 py-2 md:hidden">
          <button
            onClick={() => setSidebarOpen(true)}
            aria-label="Open chat list"
            className="rounded-lg p-1.5 text-neutral-500 hover:bg-neutral-100"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              <path d="M3 6h18M3 12h18M3 18h18" />
            </svg>
          </button>
          <span className="text-sm font-semibold">multimodal-rag</span>
        </div>

        {hasThread ? (
          <>
            <main className="flex-1 overflow-y-auto">
              <div className="mx-auto w-full max-w-4xl px-4">
                {turns.map((turn, i) => (
                  <ChatTurnView key={i} turn={turn} latest={i === turns.length - 1} />
                ))}
                <div ref={bottomRef} className="h-4" />
              </div>
            </main>
            <footer className="border-t border-neutral-200 bg-neutral-50/80 px-4 py-3 backdrop-blur">
              <div className="mx-auto flex max-w-4xl justify-center">
                <QueryBar docked busy={busy} onSubmit={ask} onClear={() => {}} />
              </div>
            </footer>
          </>
        ) : (
          <main className="flex flex-1 flex-col items-center justify-center px-4 pb-24">
            <header className="mb-8 text-center">
              <h1 className="text-2xl font-semibold tracking-tight">multimodal-rag</h1>
              <p className="mt-2 text-sm text-neutral-500">
                Grounded answers from your documents, images, and audio — cited or declined.
              </p>
            </header>
            <QueryBar docked={false} busy={busy} onSubmit={ask} onClear={() => {}} />
            <p className="mt-4 text-xs text-neutral-400">
              <kbd className="rounded border border-neutral-200 bg-white px-1">/</kbd> to focus ·{" "}
              <kbd className="rounded border border-neutral-200 bg-white px-1">Enter</kbd> to ask ·{" "}
              <kbd className="rounded border border-neutral-200 bg-white px-1">Esc</kbd> to clear
            </p>
          </main>
        )}
      </div>
    </div>
  );
}
