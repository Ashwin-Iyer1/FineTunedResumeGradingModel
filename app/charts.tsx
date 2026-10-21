"use client";

import { useState } from "react";
import type { BenchmarkRow } from "@/lib/hub";
import { cosineToScore, type Calibration } from "@/lib/model";

const fmt2 = (v: number) => v.toFixed(2);

function Tooltip({ children, style }: { children: React.ReactNode; style: React.CSSProperties }) {
  return (
    <div
      role="tooltip"
      style={style}
      className="pointer-events-none absolute z-10 w-max max-w-64 -translate-x-1/2 rounded-md border border-black/10 bg-[var(--surface-1)] px-2.5 py-1.5 text-xs shadow-lg dark:border-white/10"
    >
      {children}
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <div className="flex flex-wrap gap-4 text-xs text-[var(--ink-2)]">
      {items.map((i) => (
        <span key={i.label} className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: i.color }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}

/** Before/after dots per benchmark on one shared 0-1 axis. */
export function Dumbbell({
  rows,
  domain = [0.3, 1],
  ticks = [0.4, 0.6, 0.8, 1],
}: {
  rows: BenchmarkRow[];
  domain?: [number, number];
  ticks?: number[];
}) {
  const [hover, setHover] = useState<string | null>(null);
  const pct = (v: number) => `${((v - domain[0]) / (domain[1] - domain[0])) * 100}%`;
  return (
    <div className="flex flex-col gap-1">
      {rows.map((r) => {
        const lo = Math.min(r.baseline, r.r2);
        const hi = Math.max(r.baseline, r.r2);
        return (
          <div
            key={r.id}
            className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)] items-center gap-3 sm:grid-cols-[minmax(0,12rem)_minmax(0,1fr)]"
            onMouseEnter={() => setHover(r.id)}
            onMouseLeave={() => setHover(null)}
          >
            <div className="min-w-0 py-2">
              <p className="truncate text-sm">{r.label}</p>
              <p className="truncate text-xs text-[var(--ink-muted)]">{r.detail}</p>
            </div>
            <div className="relative h-10">
              {ticks.map((t) => (
                <span key={t} className="absolute inset-y-0 w-px bg-[var(--grid)]" style={{ left: pct(t) }} />
              ))}
              <span
                className="absolute top-1/2 h-0.5 -translate-y-1/2 rounded-full bg-[var(--axis)]"
                style={{ left: pct(lo), width: `calc(${pct(hi)} - ${pct(lo)})` }}
              />
              <span
                className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-[var(--surface-1)]"
                style={{ left: pct(r.baseline), background: "var(--baseline-mark)" }}
              />
              <span
                className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-[var(--surface-1)]"
                style={{ left: pct(r.r2), background: "var(--series-1)" }}
              />
              <span
                className="absolute top-1/2 -translate-y-1/2 pl-3 text-xs font-medium tabular-nums"
                style={{ left: pct(r.r2) }}
              >
                {fmt2(r.r2)}
              </span>
              {hover === r.id && (
                <Tooltip style={{ left: pct(r.r2), bottom: "100%" }}>
                  <p className="font-medium">{r.label}</p>
                  <p className="text-[var(--ink-2)]">
                    August baseline {fmt2(r.baseline)} → r2 {fmt2(r.r2)} ({r.r2 >= r.baseline ? "+" : ""}
                    {fmt2(r.r2 - r.baseline)})
                  </p>
                  {r.ci95 && (
                    <p className="text-[var(--ink-muted)]">
                      Paired 95% CI of change [{r.ci95[0] >= 0 ? "+" : ""}
                      {fmt2(r.ci95[0])}, {r.ci95[1] >= 0 ? "+" : ""}
                      {fmt2(r.ci95[1])}]
                    </p>
                  )}
                </Tooltip>
              )}
            </div>
          </div>
        );
      })}
      <div className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)] gap-3 sm:grid-cols-[minmax(0,12rem)_minmax(0,1fr)]">
        <span />
        <div className="relative h-4 text-[11px] tabular-nums text-[var(--ink-muted)]">
          {ticks.map((t) => (
            <span key={t} className="absolute -translate-x-1/2" style={{ left: pct(t) }}>
              {t.toFixed(1)}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

/** Distribution of 0-100 scores in 5-point bins. */
export function Histogram({ values, label }: { values: number[]; label: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const bins = Array.from({ length: 20 }, () => 0);
  for (const v of values) bins[Math.min(19, Math.floor(v / 5))]++;
  const max = Math.max(1, ...bins);
  const niceMax = max <= 10 ? 10 : Math.ceil(max / 20) * 20;
  return (
    <div>
      <div className="relative flex h-36 items-end gap-0.5 border-b border-[var(--axis)]">
        {[0.5, 1].map((f) => (
          <span key={f} className="absolute inset-x-0 h-px bg-[var(--grid)]" style={{ bottom: `${f * 100}%` }}>
            <span className="absolute -top-2 right-0 bg-[var(--surface-1)] pl-1 text-[10px] tabular-nums text-[var(--ink-muted)]">
              {Math.round(niceMax * f)}
            </span>
          </span>
        ))}
        {bins.map((count, i) => (
          <div
            key={i}
            className="relative flex h-full flex-1 items-end justify-center"
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
          >
            <div
              className="w-full max-w-6 rounded-t"
              style={{
                height: `${(count / niceMax) * 100}%`,
                minHeight: count ? 2 : 0,
                background: "var(--series-1)",
                opacity: hover === null || hover === i ? 1 : 0.55,
              }}
            />
            {hover === i && (
              <Tooltip style={{ left: "50%", bottom: `${(count / niceMax) * 100 + 4}%` }}>
                {count} {label} scored {i * 5}–{i * 5 + 4}
              </Tooltip>
            )}
          </div>
        ))}
      </div>
      <div className="mt-1 flex justify-between text-[10px] tabular-nums text-[var(--ink-muted)]">
        <span>0</span>
        <span>25</span>
        <span>50</span>
        <span>75</span>
        <span>100</span>
      </div>
    </div>
  );
}

/** The cosine -> score mapping, with a rug of where the jobs actually fell. */
export function CalibrationCurve({
  calibration,
  cosines,
  best,
}: {
  calibration: Calibration;
  cosines?: Float32Array | number[];
  best?: number;
}) {
  const [hoverX, setHoverX] = useState<number | null>(null);
  const W = 320;
  const H = 180;
  const pad = { l: 28, r: 10, t: 10, b: 26 };
  const x = (c: number) => pad.l + c * (W - pad.l - pad.r);
  const y = (s: number) => H - pad.b - (s / 100) * (H - pad.t - pad.b);
  const steps = 100;
  const path = Array.from({ length: steps + 1 }, (_, i) => {
    const c = i / steps;
    return `${i ? "L" : "M"}${x(c).toFixed(1)},${y(cosineToScore(c, calibration)).toFixed(1)}`;
  }).join(" ");

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const c = (px - pad.l) / (W - pad.l - pad.r);
    setHoverX(c >= 0 && c <= 1 ? c : null);
  };

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label="Calibration curve mapping cosine similarity to a 0 to 100 score"
        onMouseMove={onMove}
        onMouseLeave={() => setHoverX(null)}
      >
        {[0, 50, 100].map((s) => (
          <g key={s}>
            <line x1={pad.l} x2={W - pad.r} y1={y(s)} y2={y(s)} stroke="var(--grid)" strokeWidth={1} />
            <text x={pad.l - 6} y={y(s) + 3} textAnchor="end" fontSize={9} fill="var(--ink-muted)">
              {s}
            </text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((c) => (
          <text key={c} x={x(c)} y={H - pad.b + 13} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
            {c}
          </text>
        ))}
        <text x={(W + pad.l) / 2} y={H - 2} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
          cosine similarity
        </text>
        {cosines &&
          Array.from(cosines, (c, i) => (
            <line
              key={i}
              x1={x(c)}
              x2={x(c)}
              y1={H - pad.b - 7}
              y2={H - pad.b}
              stroke="var(--series-1)"
              strokeOpacity={0.25}
              strokeWidth={1}
            />
          ))}
        <path d={path} fill="none" stroke="var(--series-1)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {best !== undefined && (
          <circle
            cx={x(best)}
            cy={y(cosineToScore(best, calibration))}
            r={4.5}
            fill="var(--series-1)"
            stroke="var(--surface-1)"
            strokeWidth={2}
          />
        )}
        {hoverX !== null && (
          <g>
            <line x1={x(hoverX)} x2={x(hoverX)} y1={pad.t} y2={H - pad.b} stroke="var(--axis)" strokeWidth={1} />
            <circle
              cx={x(hoverX)}
              cy={y(cosineToScore(hoverX, calibration))}
              r={4}
              fill="var(--series-1)"
              stroke="var(--surface-1)"
              strokeWidth={2}
            />
          </g>
        )}
      </svg>
      {hoverX !== null && (
        <Tooltip style={{ left: `${(x(hoverX) / W) * 100}%`, top: 0 }}>
          cosine {hoverX.toFixed(2)} → score {cosineToScore(hoverX, calibration)}
        </Tooltip>
      )}
    </div>
  );
}
