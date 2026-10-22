"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CalibrationCurve, Histogram } from "./charts";
import { cosinesFor, loadJobIndex, type JobIndex } from "@/lib/jobs";
import { extractSkills, matchSkills, type SkillMatch } from "@/lib/matcher";
import { BUNDLED_CALIBRATION, JOB_CHAR_CAP, cosineToScore, resolveCalibration, type Calibration } from "@/lib/model";
import { extractResumeText } from "@/lib/resume";

type ModelState =
  | { status: "idle" }
  | { status: "loading"; label: string; pct?: number }
  | { status: "ready"; revision: string; threads: number; loadMs: number }
  | { status: "error"; message: string };

type Chunk = { chars: number; tokens: number; truncated: boolean; preview: string };

type Run = {
  resumeChars: number;
  chunks: Chunk[];
  embedMs: number;
  compareMs: number;
  cosines: Float32Array;
  grades: Grade[];
};

type Grade = {
  index: number;
  cosine: number;
  ai: number;
  keyword: SkillMatch;
};

const BLEND_KEY = "model-runner:ai-weight";
const PAGE = 20;

function scoreTone(score: number) {
  if (score >= 70) return "bg-emerald-600 text-white";
  if (score >= 50) return "bg-lime-500 text-zinc-950";
  if (score >= 30) return "bg-amber-400 text-zinc-950";
  return "bg-zinc-300 text-zinc-800 dark:bg-zinc-700 dark:text-zinc-100";
}

/** Owns the embedding worker and the precomputed job index. */
function useModel() {
  const workerRef = useRef<Worker | null>(null);
  const readyRef = useRef<Promise<{ index: JobIndex; calibration: Calibration }> | null>(null);
  const pending = useRef(new Map<number, { resolve: (m: { vector: Float32Array; chunks: Chunk[]; ms: number }) => void; reject: (e: Error) => void }>());
  const nextId = useRef(1);
  const [state, setState] = useState<ModelState>({ status: "idle" });

  useEffect(() => () => workerRef.current?.terminate(), []);

  const load = useCallback(() => {
    if (readyRef.current) return readyRef.current;
    readyRef.current = (async () => {
      const t0 = performance.now();
      setState({ status: "loading", label: "Loading 700 job vectors" });
      const index = await loadJobIndex();
      // Pinned to the revision the job vectors were embedded with, so the
      // resume and jobs always share one embedding space.
      const calibration = await resolveCalibration(index.revision).catch(() => BUNDLED_CALIBRATION);

      const worker = new Worker(new URL("../lib/grader.worker.ts", import.meta.url), { type: "module" });
      workerRef.current = worker;
      const threads = await new Promise<number>((resolve, reject) => {
        worker.onerror = (e) => reject(new Error(e.message || "Model worker failed."));
        worker.onmessage = (e) => {
          const msg = e.data;
          if (msg.type === "ready") resolve(msg.threads);
          else if (msg.type === "progress") {
            setState({ status: "loading", label: `Downloading ${msg.file ?? "model"}`, pct: msg.pct });
          } else if (msg.type === "embedded") {
            pending.current.get(msg.id)?.resolve(msg);
            pending.current.delete(msg.id);
          } else if (msg.type === "error") {
            const p = msg.id !== undefined ? pending.current.get(msg.id) : null;
            if (p) {
              p.reject(new Error(msg.message));
              pending.current.delete(msg.id);
            } else reject(new Error(msg.message));
          }
        };
        setState({ status: "loading", label: "Starting ONNX runtime" });
        worker.postMessage({ type: "init", revision: index.revision });
      });
      setState({ status: "ready", revision: index.revision, threads, loadMs: Math.round(performance.now() - t0) });
      return { index, calibration };
    })().catch((err: Error) => {
      readyRef.current = null;
      workerRef.current?.terminate();
      workerRef.current = null;
      setState({ status: "error", message: err.message });
      throw err;
    });
    return readyRef.current;
  }, []);

  const embed = useCallback(
    async (text: string) => {
      const ctx = await load();
      const id = nextId.current++;
      const result = await new Promise<{ vector: Float32Array; chunks: Chunk[]; ms: number }>((resolve, reject) => {
        pending.current.set(id, { resolve, reject });
        workerRef.current!.postMessage({ type: "embed", id, text });
      });
      return { ...ctx, ...result };
    },
    [load]
  );

  return { state, load, embed };
}

export default function Grader() {
  const model = useModel();
  const [resumeText, setResumeText] = useState("");
  const [resumeName, setResumeName] = useState<string | null>(null);
  const [resumeError, setResumeError] = useState<string | null>(null);
  const [parsing, setParsing] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [showResume, setShowResume] = useState(false);

  const [index, setIndex] = useState<JobIndex | null>(null);
  const [calibration, setCalibration] = useState<Calibration>(BUNDLED_CALIBRATION);
  const [run, setRun] = useState<Run | null>(null);
  const [gradedResume, setGradedResume] = useState("");
  const [grading, setGrading] = useState(false);
  const [gradeError, setGradeError] = useState<string | null>(null);

  const [aiWeight, setAiWeight] = useState(100);
  const [fn, setFn] = useState("All");
  const [level, setLevel] = useState("All");
  const [query, setQuery] = useState("");
  const [shown, setShown] = useState(PAGE);
  const [expanded, setExpanded] = useState<string | null>(null);

  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    try {
      const stored = Number(localStorage.getItem(BLEND_KEY));
      // Restore the per-browser blend after hydration so server and client match.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      if (stored >= 0 && stored <= 100 && localStorage.getItem(BLEND_KEY) !== null) setAiWeight(stored);
    } catch {
      // Storage blocked: the default blend applies.
    }
  }, []);

  const changeWeight = (w: number) => {
    setAiWeight(w);
    try {
      localStorage.setItem(BLEND_KEY, String(w));
    } catch {
      // Storage blocked: the blend just won't persist.
    }
  };

  const handleFile = useCallback(async (file: File | undefined) => {
    if (!file) return;
    setResumeError(null);
    setParsing(true);
    try {
      setResumeText(await extractResumeText(file));
      setResumeName(file.name);
    } catch (err) {
      setResumeError((err as Error).message);
    } finally {
      setParsing(false);
    }
  }, []);

  const grade = useCallback(async () => {
    if (!resumeText.trim()) return;
    setGradeError(null);
    setGrading(true);
    try {
      const { index, calibration, vector, chunks, ms } = await model.embed(resumeText);
      const t0 = performance.now();
      const cosines = cosinesFor(index, vector);
      const resumeSkills = extractSkills(resumeText);
      const grades = index.jobs.map((job, i) => ({
        index: i,
        cosine: cosines[i],
        ai: cosineToScore(cosines[i], calibration),
        keyword: matchSkills(resumeSkills, `${job.title}\n${job.description}`),
      }));
      setIndex(index);
      setCalibration(calibration);
      setRun({
        resumeChars: resumeText.length,
        chunks,
        embedMs: ms,
        compareMs: Math.round(performance.now() - t0),
        cosines,
        grades,
      });
      setGradedResume(resumeText);
      setShown(PAGE);
    } catch (err) {
      setGradeError((err as Error).message);
    } finally {
      setGrading(false);
    }
  }, [model, resumeText]);

  const ranked = useMemo(() => {
    if (!run) return [];
    const w = aiWeight / 100;
    return run.grades
      .map((g) => ({ ...g, total: Math.round(w * g.ai + (1 - w) * g.keyword.score) }))
      .sort((a, b) => b.total - a.total || b.cosine - a.cosine);
  }, [run, aiWeight]);

  const functions = useMemo(() => (index ? ["All", ...new Set(index.jobs.map((j) => j.function))].sort() : []), [index]);
  const levels = useMemo(() => (index ? ["All", ...new Set(index.jobs.map((j) => j.level))] : []), [index]);

  const filtered = useMemo(() => {
    if (!index) return [];
    const q = query.trim().toLowerCase();
    return ranked.filter((g) => {
      const job = index.jobs[g.index];
      return (
        (fn === "All" || job.function === fn) &&
        (level === "All" || job.level === level) &&
        (!q || `${job.title} ${job.company} ${job.location}`.toLowerCase().includes(q))
      );
    });
  }, [ranked, index, fn, level, query]);

  const best = ranked[0];
  const stale = run && gradedResume !== resumeText;

  return (
    <>
      <section id="try" className="scroll-mt-6">
        <SectionHeading
          eyebrow="Try it"
          title="Grade your resume against 700 real job postings"
          body="Your resume is parsed and embedded on this device. Only the model weights and the public job set are downloaded."
        />
        <ModelStatus state={model.state} onLoad={() => model.load().catch(() => {})} />

        <div className="mt-6 grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.7fr)]">
          <div className="flex flex-col gap-4">
            <Panel>
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragging(false);
                  handleFile(e.dataTransfer.files[0]);
                }}
                onClick={() => fileInput.current?.click()}
                className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-10 text-center transition-colors ${
                  dragging
                    ? "border-[var(--series-1)] bg-[var(--series-1-wash)]"
                    : "border-zinc-300 hover:border-zinc-400 dark:border-zinc-700 dark:hover:border-zinc-500"
                }`}
              >
                <p className="font-medium">{parsing ? "Reading resume…" : resumeName ?? "Drop your resume here"}</p>
                <p className="mt-1 text-sm text-zinc-500">
                  {resumeName ? "Click or drop to replace" : "PDF or .txt · never uploaded"}
                </p>
                <input
                  ref={fileInput}
                  type="file"
                  accept=".pdf,.txt,.md,application/pdf,text/plain"
                  className="hidden"
                  onChange={(e) => {
                    handleFile(e.target.files?.[0]);
                    e.target.value = "";
                  }}
                />
              </div>
              {resumeError && <p className="mt-2 text-sm text-rose-600">{resumeError}</p>}
              <button
                type="button"
                className="mt-3 text-sm text-zinc-600 underline underline-offset-2 dark:text-zinc-400"
                onClick={() => setShowResume((s) => !s)}
              >
                {showResume ? "Hide resume text" : resumeText ? "Review extracted text" : "Or paste resume text"}
              </button>
              {showResume && (
                <textarea
                  value={resumeText}
                  onChange={(e) => setResumeText(e.target.value)}
                  rows={12}
                  placeholder="Paste your resume text…"
                  className="mt-2 w-full rounded-md border border-zinc-300 bg-transparent p-3 font-mono text-xs dark:border-zinc-700"
                />
              )}
              <button
                type="button"
                disabled={grading || parsing || !resumeText.trim()}
                onClick={grade}
                className="mt-4 h-11 w-full rounded-full bg-foreground px-6 font-medium text-background transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
              >
                {grading ? "Grading…" : run ? "Re-grade resume" : "Grade my resume"}
              </button>
              {gradeError && <p className="mt-2 text-sm text-rose-600">{gradeError}</p>}
            </Panel>

            {run && (
              <Panel>
                <h3 className="text-sm font-semibold">How your resume scored across all 700 postings</h3>
                <p className="mb-3 text-xs text-zinc-500">Blended match score, {aiWeight}% AI</p>
                <Histogram values={ranked.map((g) => g.total)} label="postings" />
                <label className="mt-5 block text-xs text-zinc-600 dark:text-zinc-400">
                  <span className="flex justify-between">
                    <span>Keyword {100 - aiWeight}%</span>
                    <span>AI {aiWeight}%</span>
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={5}
                    value={aiWeight}
                    onChange={(e) => changeWeight(Number(e.target.value))}
                    aria-label="Semantic AI weight"
                    className="mt-1 w-full accent-[var(--series-1)]"
                  />
                </label>
              </Panel>
            )}
          </div>

          <Panel>
            {!run || !index ? (
              <div className="flex h-full min-h-64 flex-col items-center justify-center text-center text-zinc-500">
                <p>Upload a resume and press <span className="font-medium">Grade my resume</span>.</p>
                <p className="mt-1 text-sm">The first run downloads the 34 MB model; later runs take about a second.</p>
              </div>
            ) : (
              <>
                {stale && (
                  <p className="mb-3 rounded-md bg-amber-100 px-3 py-2 text-sm text-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
                    Your resume changed since this grade. Re-grade to update.
                  </p>
                )}
                <div className="mb-4 flex flex-wrap gap-2">
                  <Select value={fn} onChange={(v) => { setFn(v); setShown(PAGE); }} options={functions} label="Job function" />
                  <Select value={level} onChange={(v) => { setLevel(v); setShown(PAGE); }} options={levels} label="Experience level" />
                  <input
                    value={query}
                    onChange={(e) => { setQuery(e.target.value); setShown(PAGE); }}
                    placeholder="Search title, company, location"
                    className="h-9 min-w-0 flex-1 rounded-md border border-zinc-300 bg-transparent px-3 text-sm dark:border-zinc-700"
                  />
                </div>
                <p className="mb-3 text-xs text-zinc-500">
                  {filtered.length} of {index.jobs.length} postings · ranked by {100 - aiWeight}% keyword + {aiWeight}% AI
                </p>
                <ol className="flex flex-col gap-3">
                  {filtered.slice(0, shown).map((g) => {
                    const job = index.jobs[g.index];
                    const rank = ranked.indexOf(g) + 1;
                    return (
                      <li key={job.id} className="rounded-lg border border-zinc-200 p-4 dark:border-zinc-800">
                        <div className="flex items-start gap-4">
                          <div className={`flex h-14 w-14 shrink-0 flex-col items-center justify-center rounded-lg ${scoreTone(g.total)}`}>
                            <span className="text-xl font-semibold leading-none">{g.total}</span>
                            <span className="mt-0.5 text-[10px] uppercase tracking-wide opacity-80">#{rank}</span>
                          </div>
                          <div className="min-w-0 flex-1">
                            <p className="font-medium">{job.title}</p>
                            <p className="text-sm text-zinc-500">
                              {job.company}
                              {job.location && ` · ${job.location}`}
                            </p>
                            <p className="mt-1 flex flex-wrap gap-x-3 text-xs text-zinc-500">
                              <span>{job.function}</span>
                              <span>{job.level}</span>
                              <span>
                                Keyword {g.keyword.score} · AI {g.ai} <span className="opacity-60">(cos {g.cosine.toFixed(3)})</span>
                              </span>
                            </p>
                            <SkillChips match={g.keyword} />
                            <div className="mt-2 flex gap-4 text-xs">
                              <button
                                type="button"
                                className="text-zinc-600 underline underline-offset-2 dark:text-zinc-400"
                                onClick={() => setExpanded((e) => (e === job.id ? null : job.id))}
                              >
                                {expanded === job.id ? "Hide description" : "Show description"}
                              </button>
                              {job.url && (
                                <a href={job.url} target="_blank" rel="noreferrer" className="text-zinc-600 underline underline-offset-2 dark:text-zinc-400">
                                  Original posting
                                </a>
                              )}
                            </div>
                            {expanded === job.id && (
                              <p className="mt-2 whitespace-pre-wrap text-sm text-zinc-700 dark:text-zinc-300">
                                <mark className="bg-[var(--series-1-wash)] text-inherit" title="The part of the posting the model reads">
                                  {job.description.slice(0, Math.max(0, JOB_CHAR_CAP - job.title.length - 1))}
                                </mark>
                                {job.description.slice(Math.max(0, JOB_CHAR_CAP - job.title.length - 1))}
                              </p>
                            )}
                          </div>
                        </div>
                      </li>
                    );
                  })}
                </ol>
                {filtered.length > shown && (
                  <button
                    type="button"
                    onClick={() => setShown((s) => s + PAGE)}
                    className="mt-4 w-full rounded-md border border-zinc-300 py-2 text-sm hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
                  >
                    Show {Math.min(PAGE, filtered.length - shown)} more
                  </button>
                )}
                <p className="mt-4 text-xs text-zinc-500">
                  AI scores are calibrated cosine similarity, scaled so the calibration ceiling reads 100. The model reads
                  meaning, not hard requirements; check licenses, clearances, and eligibility yourself.
                </p>
              </>
            )}
          </Panel>
        </div>
      </section>

      <HowItWorks run={run} calibration={calibration} best={best?.cosine} model={model.state} />
    </>
  );
}

function HowItWorks({
  run,
  calibration,
  best,
  model,
}: {
  run: Run | null;
  calibration: Calibration;
  best?: number;
  model: ModelState;
}) {
  const tokens = run?.chunks.reduce((s, c) => s + c.tokens, 0);
  const steps: { title: string; body: string; live?: string | false | null }[] = [
    {
      title: "Extract",
      body: "pdf.js reads the PDF's text layer in your browser. Nothing is sent to a server.",
      live: run && `${run.resumeChars.toLocaleString()} characters extracted`,
    },
    {
      title: "Chunk",
      body: "The model sees 256 tokens at a time, so the resume is split into ~1,100-character chunks on paragraph and sentence boundaries instead of being truncated.",
      live: run && `${run.chunks.length} chunk${run.chunks.length === 1 ? "" : "s"} · ${tokens} tokens${run.chunks.some((c) => c.truncated) ? " · some chunks hit the 256-token cap" : ""}`,
    },
    {
      title: "Encode",
      body: "Each chunk runs through the fine-tuned BGE-small encoder (int8 ONNX, WebAssembly) in a Web Worker. Token vectors are mean-pooled and L2-normalized into one 384-dimensional embedding.",
      live:
        run &&
        `${run.embedMs} ms on ${model.status === "ready" ? model.threads : 1} thread${model.status === "ready" && model.threads > 1 ? "s" : ""}`,
    },
    {
      title: "Average",
      body: "Chunk embeddings are averaged and re-normalized, so every section of the resume counts equally.",
      live: run && `${run.chunks.length} × 384 → 1 × 384`,
    },
    {
      title: "Compare",
      body: "Every job posting (title plus its first 1,200 characters) was embedded ahead of time with the same model revision. Matching is one dot product per job.",
      live: run && `700 cosine similarities in ${run.compareMs} ms`,
    },
    {
      title: "Calibrate & blend",
      body: "A monotonic curve from calibration.json turns cosine into a 0–100 grade, scaled so the curve's ceiling reads 100. That grade is blended with a keyword skill-overlap score (0/100 by default).",
      live: run && best !== undefined && `best match: cosine ${best.toFixed(3)} → AI ${cosineToScore(best, calibration)}`,
    },
  ];

  return (
    <section id="how" className="scroll-mt-6">
      <SectionHeading
        eyebrow="How it works"
        title="From PDF to ranked matches, all in the browser"
        body={run ? "Live values are from the run you just did." : "Grade a resume above to fill in live values for each step."}
      />
      <div className="mt-6 grid gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <ol className="relative flex flex-col gap-5 border-l border-[var(--grid)] pl-6">
          {steps.map((s, i) => (
            <li key={s.title} className="relative">
              <span className="absolute -left-[37px] flex h-6 w-6 items-center justify-center rounded-full border border-[var(--grid)] bg-[var(--surface-1)] text-xs font-medium tabular-nums">
                {i + 1}
              </span>
              <p className="font-medium">{s.title}</p>
              <p className="mt-0.5 text-sm text-[var(--ink-2)]">{s.body}</p>
              {s.live && (
                <p className="mt-1.5 inline-block rounded bg-[var(--series-1-wash)] px-2 py-0.5 font-mono text-xs">{s.live}</p>
              )}
            </li>
          ))}
        </ol>
        <Panel>
          <h3 className="text-sm font-semibold">Calibration curve</h3>
          <p className="mb-3 text-xs text-zinc-500">
            Cosine below ~0.24 maps to 0; the top knot (~0.80) maps to 100.
            {run && " Ticks along the bottom are your 700 job cosines; the dot is your best match."}
          </p>
          <CalibrationCurve calibration={calibration} cosines={run?.cosines} best={best} />
          {run && (
            <div className="mt-4 border-t border-[var(--grid)] pt-3">
              <h4 className="text-xs font-semibold">Your resume&apos;s chunks</h4>
              <ul className="mt-2 flex flex-col gap-1.5">
                {run.chunks.map((c, i) => (
                  <li key={i} className="text-xs">
                    <span className="font-mono tabular-nums text-zinc-500">
                      #{i + 1} · {c.tokens} tok{c.truncated ? " (capped)" : ""}
                    </span>{" "}
                    <span className="text-[var(--ink-2)]">{c.preview}…</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Panel>
      </div>
    </section>
  );
}

function ModelStatus({ state, onLoad }: { state: ModelState; onLoad: () => void }) {
  return (
    <div className="mt-4 flex flex-wrap items-center gap-3 text-sm">
      <span
        className={`inline-block h-2.5 w-2.5 rounded-full ${
          state.status === "ready"
            ? "bg-emerald-500"
            : state.status === "loading"
              ? "animate-pulse bg-amber-400"
              : state.status === "error"
                ? "bg-rose-500"
                : "bg-zinc-400"
        }`}
      />
      {state.status === "idle" && (
        <>
          <span className="text-zinc-600 dark:text-zinc-400">Model not loaded (34 MB, cached after first use)</span>
          <SmallButton onClick={onLoad}>Load now</SmallButton>
        </>
      )}
      {state.status === "loading" && (
        <span className="text-zinc-600 dark:text-zinc-400">
          {state.label}
          {state.pct !== undefined && ` · ${state.pct}%`}
        </span>
      )}
      {state.status === "ready" && (
        <span className="text-zinc-600 dark:text-zinc-400">
          Model ready in {(state.loadMs / 1000).toFixed(1)} s · revision <code className="font-mono">{state.revision.slice(0, 8)}</code>
        </span>
      )}
      {state.status === "error" && (
        <>
          <span className="text-rose-600">Model failed to load: {state.message}</span>
          <SmallButton onClick={onLoad}>Retry</SmallButton>
        </>
      )}
    </div>
  );
}

function SkillChips({ match }: { match: SkillMatch }) {
  if (!match.matched.length && !match.implied.length && !match.missing.length) return null;
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {match.matched.map((s) => (
        <Chip key={s} title="On your resume" className="bg-emerald-100 text-emerald-900 dark:bg-emerald-950/50 dark:text-emerald-200">
          ✓ {s}
        </Chip>
      ))}
      {match.implied.map((s) => (
        <Chip key={s} title="Implied by a more specific skill on your resume" className="bg-sky-100 text-sky-900 dark:bg-sky-950/50 dark:text-sky-200">
          ≈ {s}
        </Chip>
      ))}
      {match.missing.map((s) => (
        <Chip key={s} title="In the posting, not found on your resume" className="bg-zinc-100 text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400">
          ✗ {s}
        </Chip>
      ))}
    </div>
  );
}

function Chip({ children, className, title }: { children: React.ReactNode; className: string; title?: string }) {
  return (
    <span title={title} className={`rounded-full px-2 py-0.5 text-xs ${className}`}>
      {children}
    </span>
  );
}

function Select({ value, onChange, options, label }: { value: string; onChange: (v: string) => void; options: string[]; label: string }) {
  return (
    <select
      aria-label={label}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="h-9 rounded-md border border-zinc-300 bg-[var(--surface-1)] px-2 text-sm dark:border-zinc-700"
    >
      {options.map((o) => (
        <option key={o} value={o}>
          {o === "All" ? `All ${label.toLowerCase()}s` : o}
        </option>
      ))}
    </select>
  );
}

export function SectionHeading({ eyebrow, title, body }: { eyebrow: string; title: string; body?: string }) {
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-wider text-[var(--series-1)]">{eyebrow}</p>
      <h2 className="mt-1 text-2xl font-semibold tracking-tight">{title}</h2>
      {body && <p className="mt-2 max-w-2xl text-[var(--ink-2)]">{body}</p>}
    </div>
  );
}

export function Panel({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-black/10 bg-[var(--surface-1)] p-5 dark:border-white/10">{children}</div>
  );
}

function SmallButton({ onClick, children }: { onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="rounded-md border border-zinc-300 px-2.5 py-1 text-xs font-medium transition-colors hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
    >
      {children}
    </button>
  );
}
