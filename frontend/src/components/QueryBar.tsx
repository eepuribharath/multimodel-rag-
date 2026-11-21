import { useEffect, useRef, useState } from "react";
import { ingestFile } from "../api";

interface Props {
  docked: boolean;
  busy: boolean;
  onSubmit: (question: string) => void;
  onClear: () => void;
}

export function QueryBar({ docked, busy, onSubmit, onClear }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [value, setValue] = useState("");
  const [ingestNote, setIngestNote] = useState<string | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = document.activeElement === inputRef.current;
      if (e.key === "/" && !typing) {
        e.preventDefault();
        inputRef.current?.focus();
      } else if (e.key === "Escape") {
        setValue("");
        onClear();
        inputRef.current?.blur();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClear]);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const question = value.trim();
    if (question && !busy) {
      onSubmit(question);
      setValue("");
    }
  };

  const attach = async (file: File | undefined) => {
    if (!file) return;
    setIngestNote(`Adding ${file.name}…`);
    try {
      const result = await ingestFile(file);
      setIngestNote(`Added ${result.source_id} — ${result.chunks} chunk${result.chunks === 1 ? "" : "s"}`);
    } catch (err) {
      setIngestNote((err as Error).message);
    }
  };

  return (
    <div className={docked ? "w-full max-w-3xl" : "w-full max-w-xl"}>
      <form
        onSubmit={submit}
        className="flex items-center gap-2 rounded-2xl border border-neutral-200 bg-white px-3 py-2 shadow-sm focus-within:border-indigo-400 focus-within:ring-2 focus-within:ring-indigo-100 transition"
      >
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          title="Add a document, image, or audio file to the library"
          aria-label="Add a file to the library"
          className="shrink-0 rounded-lg p-2 text-neutral-400 hover:bg-neutral-100 hover:text-neutral-600 transition"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
            <path d="M21.4 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.2-9.19a4 4 0 015.65 5.66l-9.2 9.19a2 2 0 01-2.82-2.83l8.49-8.48" />
          </svg>
        </button>
        <input
          ref={inputRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Ask your library anything…"
          aria-label="Question"
          className="min-w-0 flex-1 bg-transparent py-1.5 text-[15px] text-neutral-900 placeholder:text-neutral-400 focus:outline-none"
        />
        <button
          type="submit"
          disabled={busy || !value.trim()}
          className="shrink-0 rounded-xl bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition enabled:hover:bg-indigo-700 disabled:opacity-40"
        >
          Ask
        </button>
        <input
          ref={fileRef}
          type="file"
          className="hidden"
          accept=".md,.txt,.pdf,.png,.jpg,.jpeg,.gif,.webp,.wav,.mp3"
          onChange={(e) => { attach(e.target.files?.[0]); e.target.value = ""; }}
        />
      </form>
      {ingestNote && (
        <p className="mt-2 px-2 text-xs text-neutral-500" role="status">{ingestNote}</p>
      )}
    </div>
  );
}
