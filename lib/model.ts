// Model constants, text helpers, and score calibration shared by the page and
// the embedding worker. Mirrors the NUWorks Co-op Grader's semantic layer:
// bare resume/job text, mean pooling, cosine similarity, and a monotonic
// calibration fetched beside the weights at a pinned Hub revision.

export const MODEL_ID = "turtlecap/mdbr-leaf-mt-resume-grader";

// Roughly 1,200 characters fills the model's 256-token window for typical
// job descriptions; the tokenizer applies the exact limit for denser text.
export const JOB_CHAR_CAP = 1200;

export type Calibration =
  | {
      schema_version: 1;
      model_release?: string;
      method: "quantile_piecewise_linear";
      cosine_knots: number[];
      score_knots: number[];
    }
  | {
      schema_version: 1;
      model_release?: string;
      method: "affine";
      slope: number;
      intercept: number;
    };

// Bundled copy of calibration.json for revision d67ad62d, so a cached model
// can still score offline. Only valid for that exact revision.
export const BUNDLED_REVISION = "d67ad62deee2dfefa2f9d365b42abbbedfb88ef4";
export const BUNDLED_CALIBRATION: Calibration = {
  schema_version: 1,
  model_release: "major-agnostic-bge-2026-10-15-r2",
  method: "quantile_piecewise_linear",
  cosine_knots: [0.1776963988029566, 0.23981984925483318, 0.8022639657486932],
  score_knots: [0, 0, 0.7033154226801739],
};

const SHA_RE = /^[a-f0-9]{40}$/i;

/** Latest commit SHA of the model repo, so assets are pinned and cache-safe. */
export async function fetchLatestRevision(): Promise<string> {
  const res = await fetch(`https://huggingface.co/api/models/${MODEL_ID}`, {
    cache: "no-store",
    headers: { Accept: "application/json" },
  });
  if (!res.ok) throw new Error(`Hugging Face lookup failed: HTTP ${res.status}`);
  const sha = String((await res.json())?.sha || "").trim();
  if (!SHA_RE.test(sha)) throw new Error("Hugging Face returned an invalid revision.");
  return sha;
}

export function validateCalibration(value: unknown): Calibration {
  const v = value as Record<string, unknown> | null;
  if (!v || Number(v.schema_version) !== 1) {
    throw new Error("Unsupported calibration schema.");
  }
  if (v.method === "affine") {
    const slope = Number(v.slope);
    const intercept = Number(v.intercept);
    if (!(slope > 0) || !Number.isFinite(intercept)) {
      throw new Error("Invalid affine calibration.");
    }
    return { schema_version: 1, method: "affine", slope, intercept };
  }
  if (v.method !== "quantile_piecewise_linear") {
    throw new Error("Unsupported calibration method.");
  }
  const xs = Array.from((v.cosine_knots as number[]) || [], Number);
  const ys = Array.from((v.score_knots as number[]) || [], Number);
  if (xs.length < 2 || xs.length !== ys.length) {
    throw new Error("Calibration knots are incomplete.");
  }
  for (let i = 0; i < xs.length; i++) {
    if (!Number.isFinite(xs[i]) || !Number.isFinite(ys[i]) || ys[i] < 0 || ys[i] > 1) {
      throw new Error("Calibration knots are invalid.");
    }
    if (i > 0 && (xs[i] <= xs[i - 1] || ys[i] < ys[i - 1])) {
      throw new Error("Calibration knots must be monotonic.");
    }
  }
  return {
    schema_version: 1,
    model_release: String(v.model_release || ""),
    method: "quantile_piecewise_linear",
    cosine_knots: xs,
    score_knots: ys,
  };
}

/** calibration.json at the pinned revision, falling back to the bundled copy. */
export async function resolveCalibration(revision: string): Promise<Calibration> {
  try {
    const res = await fetch(
      `https://huggingface.co/${MODEL_ID}/resolve/${revision}/calibration.json`,
      { headers: { Accept: "application/json" } }
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return validateCalibration(await res.json());
  } catch (error) {
    if (revision === BUNDLED_REVISION) return BUNDLED_CALIBRATION;
    throw new Error(
      `Calibration for model ${revision.slice(0, 8)} is unavailable.`,
      { cause: error }
    );
  }
}

/**
 * Map a raw cosine onto a 0-100 display score. The published calibration
 * clamps at its validation ceiling (0.70 for r2); this stretches the curve so
 * that ceiling displays as 100 while keeping its shape and ranking.
 */
export function cosineToScore(cos: number, calibration: Calibration): number {
  let t: number;
  if (calibration.method === "affine") {
    t = calibration.slope * cos + calibration.intercept;
  } else {
    const xs = calibration.cosine_knots;
    const ys = calibration.score_knots;
    if (cos <= xs[0]) t = ys[0];
    else if (cos >= xs[xs.length - 1]) t = ys[ys.length - 1];
    else {
      let upper = 1;
      while (upper < xs.length && cos > xs[upper]) upper++;
      const lower = upper - 1;
      const fraction = (cos - xs[lower]) / (xs[upper] - xs[lower]);
      t = ys[lower] + fraction * (ys[upper] - ys[lower]);
    }
    const ceiling = ys[ys.length - 1];
    if (ceiling > 0) t /= ceiling;
  }
  return Math.round(Math.max(0, Math.min(1, t)) * 100);
}

/** HTML or pasted text -> plain natural text (case and punctuation kept). */
export function htmlToText(html: string): string {
  if (!html) return "";
  return String(html)
    .replace(/<\/?(?:li|br|p|div|tr|h[1-6]|ul|ol)[^>]*>/gi, "\n")
    .replace(/<[^>]*>?/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&(?:quot|ldquo|rdquo);/gi, '"')
    .replace(/&(?:apos|rsquo|lsquo);/gi, "'")
    .replace(/&#(\d+);/g, (_, d) => String.fromCharCode(+d))
    .replace(/[ \t]+/g, " ")
    .replace(/\s*\n\s*/g, "\n")
    .trim();
}

/**
 * Split long text into ~1100-char chunks so each usually fits the 256-token
 * window. PDF-extracted resumes are often one giant line, so oversized
 * paragraphs are further split on sentence/bullet boundaries.
 */
export function chunkText(text: string, target = 1100): string[] {
  const pieces: string[] = [];
  for (const para of text.split(/\n+/)) {
    if (para.length <= target) {
      pieces.push(para);
      continue;
    }
    let cur = "";
    for (const sentence of para.split(/(?<=[.!?])\s+|\s+(?=•)/)) {
      if (cur.length + sentence.length + 1 > target && cur) {
        pieces.push(cur);
        cur = "";
      }
      cur = cur ? `${cur} ${sentence}` : sentence;
    }
    if (cur) pieces.push(cur);
  }

  const out: string[] = [];
  let current = "";
  for (const piece of pieces) {
    if (current.length + piece.length + 1 > target && current) {
      out.push(current);
      current = "";
    }
    current = current ? `${current}\n${piece}` : piece;
  }
  if (current) out.push(current);
  return out.filter((c) => c.trim()).length ? out.filter((c) => c.trim()) : [text];
}
