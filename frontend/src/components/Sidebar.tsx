import type { ChatSummary } from "../api";

function relativeTime(iso: string): string {
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

interface Props {
  chats: ChatSummary[];
  activeId: number | null;
  open: boolean;
  onSelect: (id: number) => void;
  onNew: () => void;
  onDelete: (id: number) => void;
  onClose: () => void;
}

export function Sidebar({ chats, activeId, open, onSelect, onNew, onDelete, onClose }: Props) {
  return (
    <>
      {/* mobile scrim */}
      {open && (
        <div
          className="fixed inset-0 z-20 bg-black/20 md:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}
      <nav
        aria-label="Chats"
        className={
          "fixed inset-y-0 left-0 z-30 flex w-64 flex-col border-r border-neutral-200 bg-white transition-transform md:static md:translate-x-0 " +
          (open ? "translate-x-0" : "-translate-x-full")
        }
      >
        <div className="flex items-center justify-between p-3">
          <span className="px-1 text-sm font-semibold tracking-tight">multimodal-rag</span>
          <button
            onClick={onNew}
            className="rounded-lg bg-indigo-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-indigo-700"
          >
            + New chat
          </button>
        </div>
        <ul className="flex-1 space-y-0.5 overflow-y-auto p-2">
          {chats.length === 0 && (
            <li className="px-2 py-4 text-xs text-neutral-400">No chats yet.</li>
          )}
          {chats.map((chat) => (
            <li key={chat.id} className="group relative">
              <button
                onClick={() => onSelect(chat.id)}
                className={
                  "w-full rounded-lg px-2.5 py-2 pr-8 text-left transition " +
                  (chat.id === activeId ? "bg-indigo-50" : "hover:bg-neutral-100")
                }
              >
                <span
                  className={
                    "block truncate text-[13px] " +
                    (chat.id === activeId ? "font-medium text-indigo-900" : "text-neutral-700")
                  }
                >
                  {chat.title}
                </span>
                <span className="text-[11px] text-neutral-400">
                  {relativeTime(chat.updated_at)} · {chat.turns} turn{chat.turns === 1 ? "" : "s"}
                </span>
              </button>
              <button
                onClick={() => onDelete(chat.id)}
                aria-label={`Delete chat: ${chat.title}`}
                className="absolute right-1.5 top-1/2 hidden -translate-y-1/2 rounded-md p-1 text-neutral-400 hover:bg-neutral-200 hover:text-red-600 group-hover:block"
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                  <path d="M3 6h18M8 6V4a1 1 0 011-1h6a1 1 0 011 1v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6" />
                </svg>
              </button>
            </li>
          ))}
        </ul>
      </nav>
    </>
  );
}
