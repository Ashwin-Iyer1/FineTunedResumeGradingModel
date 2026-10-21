import Grader, { Panel, SectionHeading } from "./grader";
import { Dumbbell, Legend } from "./charts";
import { getModelCard, type ModelCard } from "@/lib/hub";
import { MODEL_ID } from "@/lib/model";

const REPO_URL = "https://github.com/Ashwin-Iyer1/NUWorks-Co-op-grader/tree/finetune";
const DATASET_URL = "https://www.kaggle.com/datasets/arshkon/linkedin-job-postings";

function compact(n: number) {
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(n);
}

export default async function Home() {
  const card = await getModelCard();
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-col gap-20 px-4 pb-20 pt-12 sm:px-6">
      <Hero card={card} />
      <Grader />
      {card ? <Evaluation card={card} /> : <EvaluationUnavailable />}
      <Training />
      <Credits />
    </main>
  );
}

function Hero({ card }: { card: ModelCard | null }) {
  const tiles = [
    { label: "Parameters", value: card ? `${(card.parameters / 1e6).toFixed(2)}M` : "33.36M" },
    { label: "Embedding size", value: "384-dim" },
    { label: "Context window", value: "256 tokens" },
    { label: "Download (int8 ONNX)", value: card ? `${Math.round(card.quantizedBytes / 1e6)} MB` : "34 MB" },
    ...(card
      ? [
          { label: "Hugging Face downloads", value: compact(card.downloads) },
          {
            label: "Last updated",
            value: new Date(card.lastModified).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }),
          },
        ]
      : []),
  ];
  return (
    <header>
      <p className="font-mono text-sm text-[var(--ink-muted)]">{MODEL_ID}</p>
      <h1 className="mt-2 max-w-3xl text-4xl font-semibold tracking-tight sm:text-5xl">
        A 33M-parameter resume–job matcher that runs in your browser
      </h1>
      <p className="mt-4 max-w-2xl text-lg text-[var(--ink-2)]">
        A fine-tune of {card?.baseModel ?? "BAAI/bge-small-en-v1.5"} trained to place resumes near the jobs they fit.
        It powers the semantic score in the NUWorks Co-op Grader extension. Here it ranks a resume against real LinkedIn
        postings from every field.
      </p>
      <div className="mt-6 flex flex-wrap gap-3 text-sm">
        <a href="#try" className="rounded-full bg-foreground px-5 py-2.5 font-medium text-background">
          Try it with your resume
        </a>
        <a
          href={`https://huggingface.co/${MODEL_ID}`}
          target="_blank"
          rel="noreferrer"
          className="rounded-full border border-black/15 px-5 py-2.5 font-medium dark:border-white/15"
        >
          Model card
        </a>
        <a
          href={REPO_URL}
          target="_blank"
          rel="noreferrer"
          className="rounded-full border border-black/15 px-5 py-2.5 font-medium dark:border-white/15"
        >
          Training code
        </a>
      </div>
      <dl className="mt-10 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-black/10 bg-black/10 sm:grid-cols-3 lg:grid-cols-6 dark:border-white/10 dark:bg-white/10">
        {tiles.map((t) => (
          <div key={t.label} className="bg-[var(--surface-1)] p-4">
            <dt className="text-xs text-[var(--ink-muted)]">{t.label}</dt>
            <dd className="mt-1 text-xl font-semibold">{t.value}</dd>
          </div>
        ))}
      </dl>
      {card && (
        <p className="mt-3 text-xs text-[var(--ink-muted)]">
          Release {card.release} · revision <code className="font-mono">{card.revision.slice(0, 8)}</code> · live from the
          Hugging Face Hub
        </p>
      )}
    </header>
  );
}

function Evaluation({ card }: { card: ModelCard }) {
  const real = card.spearman[0];
  const legend = [
    { label: "August baseline", color: "var(--baseline-mark)" },
    { label: "October r2 (this model)", color: "var(--series-1)" },
  ];
  return (
    <section id="evaluation" className="scroll-mt-6">
      <SectionHeading
        eyebrow="Evaluation"
        title="Better at ranking jobs for a resume; grades about the same"
        body={`Held-out tests from the model's published evaluation.json, comparing the current release against the August model it was trained from. Scores are averaged per resume, so each resume counts equally.`}
      />
      <div className="mt-6 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-black/10 bg-black/10 lg:grid-cols-4 dark:border-white/10 dark:bg-white/10">
        <Stat
          label="Spearman, real resumes"
          value={real.r2.toFixed(3)}
          delta={`+${(real.r2 - real.baseline).toFixed(3)} vs baseline`}
        />
        <Stat
          label="Pairwise accuracy, real resumes"
          value={`${(card.pairwise.r2 * 100).toFixed(1)}%`}
          delta={`+${((card.pairwise.r2 - card.pairwise.baseline) * 100).toFixed(1)} pts`}
        />
        <Stat
          label="NDCG@10, real resumes"
          value={card.ndcg[0].r2.toFixed(3)}
          delta={`+${(card.ndcg[0].r2 - card.ndcg[0].baseline).toFixed(3)}`}
        />
        <Stat
          label="Held-out test set"
          value={`${card.realHoldout.pairs} pairs`}
          delta={`${card.realHoldout.resumes} resumes · ${card.realHoldout.categories} occupations`}
          neutral
        />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Panel>
          <h3 className="text-sm font-semibold">Ranking quality: per-resume Spearman</h3>
          <p className="mb-3 text-xs text-[var(--ink-muted)]">How well each resume&apos;s job ordering matches the labels. Higher is better.</p>
          <Legend items={legend} />
          <div className="mt-3">
            <Dumbbell rows={card.spearman} />
          </div>
        </Panel>
        <Panel>
          <h3 className="text-sm font-semibold">Top of the list: NDCG@10</h3>
          <p className="mb-3 text-xs text-[var(--ink-muted)]">Whether the best-fitting jobs land in each resume&apos;s top 10. Higher is better.</p>
          <Legend items={legend} />
          <div className="mt-3">
            <Dumbbell rows={card.ndcg} domain={[0.7, 1]} ticks={[0.7, 0.8, 0.9, 1]} />
          </div>
        </Panel>
      </div>

      <Panel>
        <div className="mt-0">
          <h3 className="text-sm font-semibold">Absolute grades: mean absolute error, 0–100 points</h3>
          <p className="mb-3 text-xs text-[var(--ink-muted)]">
            How far the calibrated grade is from the label. Lower is better. Ranking improved more than grading did.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[28rem] text-sm">
              <thead>
                <tr className="border-b border-[var(--grid)] text-left text-xs text-[var(--ink-muted)]">
                  <th className="py-2 font-medium">Test set</th>
                  <th className="py-2 text-right font-medium">Baseline</th>
                  <th className="py-2 text-right font-medium">r2</th>
                  <th className="py-2 text-right font-medium">Change</th>
                </tr>
              </thead>
              <tbody className="tabular-nums">
                {card.grades.map((g) => {
                  const d = g.r2 - g.baseline;
                  return (
                    <tr key={g.label} className="border-b border-[var(--grid)] last:border-0">
                      <td className="py-2">{g.label}</td>
                      <td className="py-2 text-right">{g.baseline.toFixed(1)}</td>
                      <td className="py-2 text-right">{g.r2.toFixed(1)}</td>
                      <td className="py-2 text-right">
                        <span className={d < 0 ? "text-emerald-700 dark:text-emerald-400" : "text-[var(--ink-2)]"}>
                          {d < 0 ? "▼" : "▲"} {Math.abs(d).toFixed(1)} {d < 0 ? "better" : "worse"}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-[var(--ink-muted)]">
            Labels are LLM teacher judgments, not hiring outcomes. The real-resume test&apos;s 40 jobs also appear in
            older training data. Top-1 hit rate on real resumes moved {(card.top1.baseline * 100).toFixed(0)}% →{" "}
            {(card.top1.r2 * 100).toFixed(0)}%, with a confidence interval that includes zero.{" "}
            <a href={card.reportUrl} target="_blank" rel="noreferrer" className="underline underline-offset-2">
              Full report
            </a>
          </p>
        </div>
      </Panel>
    </section>
  );
}

function EvaluationUnavailable() {
  return (
    <section id="evaluation">
      <SectionHeading eyebrow="Evaluation" title="Evaluation unavailable" />
      <p className="mt-2 text-[var(--ink-2)]">
        The Hugging Face Hub couldn&apos;t be reached when this page was built. See the{" "}
        <a href={`https://huggingface.co/${MODEL_ID}`} className="underline underline-offset-2">
          model card
        </a>{" "}
        for metrics.
      </p>
    </section>
  );
}

function Stat({ label, value, delta, neutral }: { label: string; value: string; delta: string; neutral?: boolean }) {
  return (
    <div className="bg-[var(--surface-1)] p-4">
      <p className="text-xs text-[var(--ink-muted)]">{label}</p>
      <p className="mt-1 text-2xl font-semibold">{value}</p>
      <p className={`mt-0.5 text-xs ${neutral ? "text-[var(--ink-muted)]" : "text-emerald-700 dark:text-emerald-400"}`}>
        {!neutral && "▲ "}
        {delta}
      </p>
    </div>
  );
}

function Training() {
  const facts = [
    ["Base model", "BAAI/bge-small-en-v1.5, a BERT encoder (12 layers, 384 hidden)"],
    ["Objective", "CoSENT loss on teacher-labeled resume/job pairs: it teaches the ordering of cosine similarities to agree with the ordering of match scores"],
    ["Schedule", "2 epochs, batch size 64, batches grouped so each one compares jobs for the same resume"],
    ["New data in r2", "4,608 labeled pairs from 864 fictional resumes (24 majors × 6 career stages) and 144 job scenarios"],
    ["Replayed data", "Earlier Hugging Face, OpenAI-labeled, and NeuralFrame resume/job pairs, to keep earlier gains"],
    ["Selection", "Checkpoint b_epoch2, chosen on validation only; test sets opened once"],
    ["Calibration", "Validation-only piecewise-linear map, 50/50 across synthetic and legacy sources"],
    ["Inference", "No query prompt, mean pooling, L2 normalization, 256-token window, int8 ONNX for the browser"],
  ];
  return (
    <section id="training" className="scroll-mt-6">
      <SectionHeading
        eyebrow="Training"
        title="Built to be major-agnostic"
        body="The r2 release extends coverage across 24 academic majors and 6 career stages. Major and year are only sampling metadata; demonstrated skills and the job's explicit requirements decide the fit."
      />
      <dl className="mt-6 divide-y divide-[var(--grid)] rounded-xl border border-black/10 bg-[var(--surface-1)] dark:border-white/10">
        {facts.map(([k, v]) => (
          <div key={k} className="grid gap-1 px-5 py-3 sm:grid-cols-[12rem_minmax(0,1fr)]">
            <dt className="text-sm font-medium">{k}</dt>
            <dd className="text-sm text-[var(--ink-2)]">{v}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function Credits() {
  return (
    <footer className="border-t border-[var(--grid)] pt-6 text-xs text-[var(--ink-muted)]">
      <p>
        Job postings: a stratified sample of 700 (20 per LinkedIn job function) from{" "}
        <a href={DATASET_URL} target="_blank" rel="noreferrer" className="underline underline-offset-2">
          LinkedIn Job Postings 2023–2024
        </a>{" "}
        by Arsh Koneru on Kaggle, licensed CC BY-SA 4.0. Descriptions are shortened for display; the sampled set is
        shared under the same license.
      </p>
      <p className="mt-2">
        Model and training code by Ashwin Iyer:{" "}
        <a href={`https://huggingface.co/${MODEL_ID}`} target="_blank" rel="noreferrer" className="underline underline-offset-2">
          Hugging Face
        </a>{" "}
        ·{" "}
        <a href={REPO_URL} target="_blank" rel="noreferrer" className="underline underline-offset-2">
          GitHub
        </a>
        . Scores are a demo of semantic similarity, not hiring advice.
      </p>
    </footer>
  );
}
