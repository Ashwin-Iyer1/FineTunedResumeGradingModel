// Precompute job embeddings with the same q8 ONNX weights the browser loads,
// so grading only needs to embed the resume. Output is a raw little-endian
// Float32 matrix (jobs x dim) plus the revision it is valid for.
//
//   node scripts/embed_jobs.mjs
import { readFile, writeFile } from "node:fs/promises";
import { pipeline } from "@huggingface/transformers";

const MODEL_ID = "turtlecap/mdbr-leaf-mt-resume-grader";
const JOB_CHAR_CAP = 1200;
// Ordered batches of 16, matching the model card's evaluation setup.
const BATCH = 16;
const DATA = new URL("../public/data/", import.meta.url);

const meta = await (
  await fetch(`https://huggingface.co/api/models/${MODEL_ID}`)
).json();
const revision = meta.sha;
const { jobs } = JSON.parse(await readFile(new URL("jobs.json", DATA), "utf8"));

const extractor = await pipeline("feature-extraction", MODEL_ID, {
  dtype: "q8",
  revision,
  use_external_data_format: false,
});

let dim = 0;
let matrix;
const t0 = performance.now();
for (let start = 0; start < jobs.length; start += BATCH) {
  const batch = jobs
    .slice(start, start + BATCH)
    .map((j) => `${j.title}\n${j.description}`.slice(0, JOB_CHAR_CAP));
  const out = await extractor(batch, { pooling: "mean", normalize: true });
  if (!matrix) {
    dim = out.dims.at(-1);
    matrix = new Float32Array(jobs.length * dim);
  }
  matrix.set(out.data, start * dim);
  process.stdout.write(`\r${Math.min(start + BATCH, jobs.length)}/${jobs.length}`);
}
const ms = performance.now() - t0;

await writeFile(new URL("job-embeddings.bin", DATA), Buffer.from(matrix.buffer));
await writeFile(
  new URL("job-embeddings.json", DATA),
  JSON.stringify({ model: MODEL_ID, revision, dtype: "q8", dim, count: jobs.length, jobCharCap: JOB_CHAR_CAP }, null, 2)
);
console.log(`\nEmbedded ${jobs.length} jobs (${dim}-dim) at ${revision.slice(0, 8)} in ${(ms / 1000).toFixed(1)}s`);
