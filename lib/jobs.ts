// Job set sampled from the Kaggle LinkedIn Job Postings dataset by
// scripts/build_jobs.py, with vectors precomputed by scripts/embed_jobs.mjs.

export type Job = {
  id: string;
  title: string;
  company: string;
  location: string;
  function: string;
  level: string;
  workType: string;
  url: string;
  description: string;
};

export type JobSource = {
  name: string;
  url: string;
  license: string;
  totalPostings: number;
  sampling: string;
};

export type JobIndex = {
  source: JobSource;
  jobs: Job[];
  /** Row-major (jobs x dim), L2-normalized. */
  vectors: Float32Array;
  dim: number;
  revision: string;
};

export async function loadJobIndex(): Promise<JobIndex> {
  const [data, meta, bin] = await Promise.all([
    fetch("/data/jobs.json").then((r) => r.json()),
    fetch("/data/job-embeddings.json").then((r) => r.json()),
    fetch("/data/job-embeddings.bin").then((r) => r.arrayBuffer()),
  ]);
  const vectors = new Float32Array(bin);
  if (meta.count !== data.jobs.length || vectors.length !== meta.count * meta.dim) {
    throw new Error("Job embeddings are out of date; re-run scripts/embed_jobs.mjs.");
  }
  return { source: data.source, jobs: data.jobs, vectors, dim: meta.dim, revision: meta.revision };
}

/** Cosine of a normalized query against every job vector. */
export function cosinesFor(index: JobIndex, query: Float32Array): Float32Array {
  const { vectors, dim, jobs } = index;
  const out = new Float32Array(jobs.length);
  for (let j = 0; j < jobs.length; j++) {
    let sum = 0;
    const off = j * dim;
    for (let i = 0; i < dim; i++) sum += query[i] * vectors[off + i];
    out[j] = sum;
  }
  return out;
}
