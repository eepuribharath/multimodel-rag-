import { useEffect, useRef, useState } from "react";
import type { EvidencePayload } from "../api";
import { mediaUrl } from "../api";

const MODALITY_ICONS: Record<string, string> = {
  text: "¶",
  image: "▣",
  audio: "♪",
  video: "▶",
  pdf_page: "⎘",
};

function ModalityBadge({ modality }: { modality: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md bg-neutral-100 px-1.5 py-0.5 text-[11px] font-medium text-neutral-600">
      <span aria-hidden="true">{MODALITY_ICONS[modality] ?? "•"}</span>
      {modality}
    </span>
  );
}

interface Props {
  evidence: EvidencePayload;
  citedIds: string[];
  activeMarker: number | null; // 1-based index into evidence.items
  onCardHover: (n: number | null) => void;
}

export function EvidencePanel({ evidence, citedIds, activeMarker, onCardHover }: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const [inspector, setInspector] = useState(false);
  const refs = useRef<(HTMLLIElement | null)[]>([]);

  useEffect(() => {
    if (activeMarker !== null)
      refs.current[activeMarker - 1]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [activeMarker]);

  const cited = new Set(citedIds);

  return (
    <aside className="w-full">
      <div className="flex items-center justify-between">
        <button
          onClick={() => setCollapsed(!collapsed)}
          aria-expanded={!collapsed}
          className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-neutral-500 hover:text-neutral-700"
        >
          <span className={"transition-transform " + (collapsed ? "-rotate-90" : "")} aria-hidden="true">▾</span>
          Evidence ({evidence.items.length})
        </button>
        <label className="flex cursor-pointer items-center gap-1.5 text-[11px] text-neutral-400 hover:text-neutral-600">
          <input
            type="checkbox"
            checked={inspector}
            onChange={(e) => setInspector(e.target.checked)}
            className="h-3 w-3 accent-indigo-600"
          />
          retrieval inspector
        </label>
      </div>

      {!collapsed && (
        <ul className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {evidence.items.map((item, i) => {
            const n = i + 1;
            const isActive = activeMarker === n;
            const isCited = cited.has(item.chunk_id);
            return (
              <li
                key={item.chunk_id}
                ref={(el) => { refs.current[i] = el; }}
                onMouseEnter={() => onCardHover(n)}
                onMouseLeave={() => onCardHover(null)}
                className={
                  "rounded-xl border p-3.5 transition flex flex-col justify-between " +
                  (isActive
                    ? "border-indigo-400 bg-indigo-50/60 ring-2 ring-indigo-100"
                    : isCited
                      ? "border-neutral-200 bg-white shadow-xs"
                      : "border-neutral-100 bg-neutral-50/60 opacity-80")
                }
              >
                <div>
                  <div className="flex items-center gap-2">
                    <span className="inline-flex h-5 w-5 items-center justify-center rounded-md bg-indigo-600/90 text-[11px] font-bold text-white">
                      {n}
                    </span>
                    <span className="truncate text-sm font-medium text-neutral-800">{item.source_id}</span>
                    <span className="text-xs text-neutral-400">{item.locator}</span>
                    <span className="ml-auto shrink-0"><ModalityBadge modality={item.modality} /></span>
                  </div>
                  {item.snippet && (
                    <p className="mt-2 line-clamp-3 text-[13px] leading-5 text-neutral-600">{item.snippet}</p>
                  )}
                  {item.media_url && item.modality === "image" && (
                    <a
                      href={mediaUrl(item.media_url)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="mt-2.5 block overflow-hidden rounded-lg border border-neutral-200 bg-neutral-100"
                      title="Click to view full image"
                    >
                      <img
                        src={mediaUrl(item.media_url)}
                        alt={item.snippet || `figure from ${item.source_id}`}
                        className="max-h-48 w-full object-cover transition duration-150 hover:scale-[1.02]"
                      />
                    </a>
                  )}
                  {item.media_url && item.modality === "audio" && (
                    <audio
                      controls
                      preload="metadata"
                      className="mt-2.5 w-full"
                      src={mediaUrl(item.media_url)}
                      onLoadedMetadata={(e) => {
                        if (item.start_ms) e.currentTarget.currentTime = 0; // slices start at the cited timestamp already
                      }}
                    />
                  )}
                </div>
                {isCited && (
                  <p className="mt-3 text-[11px] font-semibold text-indigo-600">✓ cited in answer</p>
                )}
              </li>
            );
          })}
        </ul>
      )}

      {!collapsed && inspector && (
        <div className="mt-4 rounded-xl border border-neutral-200 bg-white p-3">
          <p className="text-[11px] font-semibold uppercase tracking-wide text-neutral-400">
            candidates: pre-rerank (top 10) → post-rerank (top {evidence.items.length})
          </p>
          <div className="mt-2 grid grid-cols-2 gap-3 font-mono text-[11px] text-neutral-600">
            <ol className="space-y-1">
              {evidence.pre_rerank.map((row, i) => (
                <li key={row.chunk_id} className="truncate">
                  {i + 1}. {row.source_id} <span className="text-neutral-400">rrf {row.rrf_score.toFixed(4)}</span>
                </li>
              ))}
            </ol>
            <ol className="space-y-1">
              {evidence.items.map((item, i) => (
                <li key={item.chunk_id} className="truncate">
                  {i + 1}. {item.source_id}{" "}
                  <span className="text-neutral-400">
                    ce {item.rerank_score?.toFixed(3) ?? "—"}
                  </span>
                </li>
              ))}
            </ol>
          </div>
        </div>
      )}
    </aside>
  );
}
