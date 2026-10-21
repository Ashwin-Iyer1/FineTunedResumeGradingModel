# Model Runner: resume–job matcher showcase

A showcase for [`turtlecap/mdbr-leaf-mt-resume-grader`](https://huggingface.co/turtlecap/mdbr-leaf-mt-resume-grader),
a 33M-parameter fine-tune of `BAAI/bge-small-en-v1.5` that embeds resumes and jobs
into a shared space. The page:

- shows live model stats and the published evaluation (baseline vs r2) from the Hub
- grades an uploaded resume against 700 real LinkedIn postings, in the browser
- walks through each step of the pipeline with the numbers from your own run

```bash
npm install
npm run dev   # http://localhost:3000
```

## How grading works

1. **Extract**: pdf.js reads the resume in the browser. It is never uploaded.
2. **Chunk**: ~1,100-character chunks, so nothing is lost to the 256-token window.
3. **Encode**: Transformers.js runs the int8 ONNX model (34 MB, cached after the
   first run) in a Web Worker, using mean pooling and L2 normalization.
4. **Average**: chunk vectors are averaged into one 384-dim resume vector.
5. **Compare**: cosine against 700 job vectors that were precomputed with the same
   model revision, so a run costs one resume embedding plus 700 dot products.
6. **Calibrate & blend**: `calibration.json` from the Hub maps cosine to 0–100,
   stretched so its 0.70 ceiling reads 100. The result is blended 65/35 with a
   keyword skill-overlap score (adjustable).

## Data

Jobs come from [LinkedIn Job Postings 2023–2024](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
(Arsh Koneru, **CC BY-SA 4.0**). `public/data/jobs.json` holds a stratified sample of 20 postings for each of LinkedIn's 35 job functions,
shared under the same license.

```bash
npm run data:jobs    # kagglehub download (no account needed) + sample → public/data/jobs.json
npm run data:embed   # embed with the latest model revision → public/data/job-embeddings.{bin,json}
```

Re-run `data:embed` after publishing a new model revision. The page pins the browser
model to the revision recorded in `job-embeddings.json`, so resume and job vectors
always come from the same weights.

## Layout

- `app/page.tsx`: hero stats, evaluation, training, and credits (server; Hub data cached hourly)
- `app/grader.tsx`: upload, grading, results, and the live how-it-works section
- `app/charts.tsx`: dumbbell, histogram, and calibration-curve charts
- `lib/hub.ts`: Hub metadata and `evaluation.json` parsing
- `lib/grader.worker.ts`: model loading and resume embedding
- `lib/jobs.ts`: job index loading and cosine scoring
- `lib/model.ts`: revision/calibration lookup, chunking, cosine→score
- `lib/matcher.ts`: skill extraction and keyword scoring
- `scripts/`: dataset sampling and job embedding
