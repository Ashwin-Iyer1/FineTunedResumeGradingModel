// Server-side model card data: live Hub metadata plus the published
// evaluation.json, cached so the page stays static between refreshes.
import { cacheLife } from "next/cache";
import { MODEL_ID } from "./model";

type Ranking = { mean: number; ci95?: [number, number] };

export type BenchmarkRow = {
  id: string;
  label: string;
  detail: string;
  baseline: number;
  r2: number;
  ci95?: [number, number];
};

export type GradeRow = { label: string; baseline: number; r2: number };

export type ModelCard = {
  revision: string;
  lastModified: string;
  downloads: number;
  likes: number;
  parameters: number;
  quantizedBytes: number;
  release: string;
  baseModel: string;
  reportUrl: string;
  spearman: BenchmarkRow[];
  ndcg: BenchmarkRow[];
  pairwise: BenchmarkRow;
  top1: BenchmarkRow;
  grades: GradeRow[];
  realHoldout: { pairs: number; resumes: number; jobs: number; categories: number };
};

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

/* eslint-disable @typescript-eslint/no-explicit-any */
export async function getModelCard(): Promise<ModelCard | null> {
  "use cache";
  cacheLife("hours");
  try {
    const meta = await getJson<any>(`https://huggingface.co/api/models/${MODEL_ID}`);
    const base = `https://huggingface.co/${MODEL_ID}/resolve/${meta.sha}`;
    const [ev, manifest] = await Promise.all([
      getJson<any>(`${base}/evaluation.json`),
      getJson<any>(`${base}/publication_manifest.json`),
    ]);

    const test = ev.original_weight_selection.test;
    const real = ev.real_holdout_current_rubric;
    const legacy = ev.real_holdout_legacy_rubric;
    const realGain = real.paired_improvement.raw_ranking_candidate_minus_baseline;
    const mv = (side: "baseline" | "selected", set: string) => test[side][set].mean_vector;
    const rank = (side: any, key: string): Ranking => side.macro_raw_ranking[key];

    const realRow = (key: string, label: string): BenchmarkRow => ({
      id: `real-${key}`,
      label: "Real resumes (held out)",
      detail: `${real.resumes} archive resumes × ${real.jobs} jobs · ${label}`,
      baseline: rank(real.baseline, key).mean,
      r2: rank(real.candidate, key).mean,
      ci95: realGain[key]?.ci95,
    });
    const testRow = (set: string, label: string, detail: string, key: string): BenchmarkRow => ({
      id: `${set}-${key}`,
      label,
      detail,
      baseline: mv("baseline", set)[key],
      r2: mv("selected", set)[key],
    });

    return {
      revision: meta.sha,
      lastModified: meta.lastModified,
      downloads: meta.downloads ?? 0,
      likes: meta.likes ?? 0,
      parameters: meta.safetensors?.total ?? 0,
      quantizedBytes: manifest.files?.["onnx/model_quantized.onnx"]?.bytes ?? 0,
      release: ev.release,
      baseModel: meta.cardData?.base_model ?? "BAAI/bge-small-en-v1.5",
      reportUrl: ev.report_url,
      spearman: [
        realRow("spearman", "current rubric"),
        testRow("diverse", "Diverse fictional test", "144 resumes, 24 majors", "mean_resume_spearman"),
        testRow("hf", "Hugging Face test", "46 rankable resumes", "mean_resume_spearman"),
      ],
      ndcg: [
        realRow("ndcg_at_10", "current rubric"),
        testRow("diverse", "Diverse fictional test", "144 resumes, 24 majors", "mean_ndcg_at_10"),
        testRow("hf", "Hugging Face test", "46 rankable resumes", "mean_ndcg_at_10"),
      ],
      pairwise: realRow("pairwise_accuracy", "pairwise accuracy"),
      top1: realRow("top1_best_rate", "top-1 hit rate"),
      grades: [
        {
          label: "Real resumes, current rubric",
          baseline: real.baseline.equal_resume_grading.display_mae_points.mean,
          r2: real.candidate.equal_resume_grading.display_mae_points.mean,
        },
        {
          label: "Real resumes, legacy rubric",
          baseline: legacy.baseline.equal_resume_grading.display_mae_points.mean,
          r2: legacy.candidate.equal_resume_grading.display_mae_points.mean,
        },
        {
          label: "Diverse fictional test",
          baseline: ev.calibration_recheck.datasets.diverse.models.baseline.equal_resume_grade_mae_points,
          r2: ev.calibration_recheck.datasets.diverse.models.selected_r2.equal_resume_grade_mae_points,
        },
        {
          label: "Default 65/35 keyword blend",
          baseline: ev.default_ui_blend_simulation.metrics.baseline.mae_points,
          r2: ev.default_ui_blend_simulation.metrics.candidate.mae_points,
        },
      ],
      realHoldout: {
        pairs: real.pairs,
        resumes: real.resumes,
        jobs: real.jobs,
        categories: ev.protocol?.holdout?.occupational_categories ?? real.resumes,
      },
    };
  } catch (error) {
    console.error("Model card unavailable:", error);
    return null;
  }
}
