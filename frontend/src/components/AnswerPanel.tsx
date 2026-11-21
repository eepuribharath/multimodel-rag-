import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Stage } from "../api";

const STAGES: Stage[] = ["retrieving", "reranking", "generating"];

export function StageProgress({ stage }: { stage: Stage }) {
  const reached = STAGES.indexOf(stage);
  return (
    <div className="flex items-center gap-3" aria-hidden="true">
      {STAGES.map((name, i) => (
        <div key={name} className="flex items-center gap-1.5">
          <span
            className={
              "h-1.5 w-1.5 rounded-full transition " +
              (i < reached ? "bg-indigo-600" :
               i === reached ? "bg-indigo-600 stage-active" : "bg-neutral-300")
            }
          />
          <span className={"text-xs " + (i <= reached ? "text-neutral-700" : "text-neutral-400")}>
            {name}
          </span>
        </div>
      ))}
    </div>
  );
}

interface AnswerTextProps {
  text: string;
  streaming: boolean;
  activeMarker: number | null;
  onMarkerHover: (n: number | null) => void;
  onMarkerClick: (n: number) => void;
}

/** Renders answer text with markdown parsing (bold, italic, lists, headers) and [n] markers as interactive citation chips. */
export function AnswerText({ text, streaming, activeMarker, onMarkerHover, onMarkerClick }: AnswerTextProps) {
  // Convert citation markers [n] into custom markdown links [n](#cite-n) so the markdown AST cleanly nests them
  const processedText = text.replace(/\[(\d+)\]/g, "[$1](#cite-$1)");

  return (
    <div
      className={"text-[15px] leading-7 text-neutral-800 space-y-2.5" + (streaming ? " caret" : "")}
      aria-live="polite"
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => (
            <p className="my-1.5 leading-7 text-neutral-800">{children}</p>
          ),
          h1: ({ children }) => (
            <h1 className="mt-4 mb-2 text-xl font-bold tracking-tight text-neutral-900 border-b border-neutral-100 pb-1">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="mt-3.5 mb-1.5 text-lg font-bold tracking-tight text-neutral-900">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="mt-3 mb-1 text-base font-semibold text-neutral-900">
              {children}
            </h3>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-neutral-900">
              {children}
            </strong>
          ),
          em: ({ children }) => (
            <em className="italic text-neutral-800">{children}</em>
          ),
          ul: ({ children }) => (
            <ul className="my-2 list-disc pl-5 space-y-1 text-neutral-800">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="my-2 list-decimal pl-5 space-y-1 text-neutral-800">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="leading-6">{children}</li>
          ),
          blockquote: ({ children }) => (
            <blockquote className="my-2.5 border-l-3 border-indigo-400 bg-neutral-50/70 py-1 pl-3.5 text-neutral-700 italic rounded-r">
              {children}
            </blockquote>
          ),
          pre: ({ children, ...props }) => (
            <pre
              className="my-3 overflow-x-auto rounded-lg bg-neutral-900 p-3.5 font-mono text-[13px] text-neutral-100 shadow-xs [&>code]:bg-transparent [&>code]:p-0 [&>code]:text-neutral-100"
              {...props}
            >
              {children}
            </pre>
          ),
          code: ({ className, children, ...props }) => (
            <code
              className={
                className ??
                "rounded bg-neutral-100 px-1.5 py-0.5 font-mono text-[13px] font-medium text-indigo-700"
              }
              {...props}
            >
              {children}
            </code>
          ),
          table: ({ children }) => (
            <div className="my-3 overflow-x-auto">
              <table className="min-w-full divide-y divide-neutral-200 border border-neutral-200 text-sm">
                {children}
              </table>
            </div>
          ),
          th: ({ children }) => (
            <th className="bg-neutral-50 px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-neutral-600">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-3 py-2 text-neutral-700 border-t border-neutral-100">
              {children}
            </td>
          ),
          a: ({ href, children }) => {
            if (href?.startsWith("#cite-")) {
              const n = Number(href.slice(6));
              if (!Number.isNaN(n)) {
                return (
                  <button
                    type="button"
                    onMouseEnter={() => onMarkerHover(n)}
                    onMouseLeave={() => onMarkerHover(null)}
                    onClick={() => onMarkerClick(n)}
                    aria-label={`Citation ${n}`}
                    className={
                      "mx-0.5 inline-flex -translate-y-0.5 items-center justify-center rounded-md px-1.5 py-0.2 text-[11px] font-semibold transition cursor-pointer select-none " +
                      (activeMarker === n
                        ? "bg-indigo-600 text-white shadow-xs"
                        : "bg-indigo-50 text-indigo-700 hover:bg-indigo-100 hover:text-indigo-900")
                    }
                  >
                    {n}
                  </button>
                );
              }
            }
            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium text-indigo-600 underline hover:text-indigo-800"
              >
                {children}
              </a>
            );
          },
        }}
      >
        {processedText}
      </ReactMarkdown>
    </div>
  );
}

export function Abstained() {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-neutral-200 bg-neutral-50 p-4">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="mt-0.5 shrink-0 text-neutral-400" aria-hidden="true">
        <circle cx="12" cy="12" r="10" /><path d="M12 8v4M12 16h.01" />
      </svg>
      <div>
        <p className="text-sm font-medium text-neutral-800">Not enough in the sources to answer</p>
        <p className="mt-1 text-sm text-neutral-500">
          The library doesn't contain material that supports an answer to this question,
          so nothing was made up.
        </p>
      </div>
    </div>
  );
}

export function EmptyResult() {
  return (
    <div className="rounded-xl border border-neutral-200 bg-neutral-50 p-4">
      <p className="text-sm font-medium text-neutral-800">Nothing matched</p>
      <p className="mt-1 text-sm text-neutral-500">
        No documents in the library match this query. Add files with the paperclip.
      </p>
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-red-200 bg-red-50 p-4" role="alert">
      <p className="text-sm font-medium text-red-800">Something went wrong</p>
      <p className="mt-1 text-sm text-red-600">{message}</p>
    </div>
  );
}
