"""Sample a diverse job set from the Kaggle LinkedIn Job Postings dataset.

Source: kaggle.com/datasets/arshkon/linkedin-job-postings (CC BY-SA 4.0),
downloaded anonymously via kagglehub (no Kaggle account needed). Postings are
stratified by LinkedIn job function so the showcase covers every field, not
just tech. Output: public/data/jobs.json (shared under the same license).

    python scripts/build_jobs.py [--per-function 20] [--seed 7]
"""

import argparse
import json
import re
from pathlib import Path

import kagglehub
import pandas as pd

DATASET = "arshkon/linkedin-job-postings"
OUT = Path(__file__).resolve().parent.parent / "public" / "data" / "jobs.json"
# Kept for display; the model only reads the first 1,200 characters.
DISPLAY_CHARS = 2400


def read_csv(name: str) -> pd.DataFrame:
    path = Path(kagglehub.dataset_download(DATASET, path=name))
    # Some files are served as zip archives despite the .csv name.
    with open(path, "rb") as f:
        zipped = f.read(2) == b"PK"
    return pd.read_csv(path, compression="zip" if zipped else None)


def clean(text: str) -> str:
    text = re.sub(r"[ \t ]+", " ", str(text))
    text = re.sub(r"\s*\n\s*", "\n", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return cut[: max(cut.rfind(" "), limit - 40)].rstrip() + "…"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-function", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    postings = read_csv("postings.csv")
    functions = read_csv("mappings/skills.csv")
    job_skills = read_csv("jobs/job_skills.csv")

    total_postings = len(postings)
    postings = postings.dropna(subset=["title", "description", "company_name"])
    postings["description"] = postings["description"].map(clean)
    # Drop stubs and the long tail of very long postings.
    lengths = postings["description"].str.len()
    postings = postings[(lengths >= 600) & (lengths <= 9000)]
    # Mostly-ASCII text keeps the set English; the model is English-only.
    ascii_share = postings["description"].map(
        lambda t: sum(c.isascii() for c in t) / len(t)
    )
    postings = postings[ascii_share > 0.97]
    postings = postings.drop_duplicates(subset=["title", "company_name"])
    postings = postings.drop_duplicates(subset=["description"])

    # A posting's first listed function is its primary one.
    primary = (
        job_skills.drop_duplicates("job_id")
        .merge(functions, on="skill_abr")
        .set_index("job_id")["skill_name"]
    )
    postings = postings[postings["job_id"].isin(primary.index)].copy()
    postings["function"] = postings["job_id"].map(primary)

    sampled = (
        postings.groupby("function", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), args.per_function), random_state=args.seed))
        .sort_values(["function", "title"])
    )

    jobs = []
    for row in sampled.itertuples():
        level = row.formatted_experience_level
        jobs.append(
            {
                "id": str(row.job_id),
                "title": clean(row.title),
                "company": clean(row.company_name),
                "location": clean(row.location) if isinstance(row.location, str) else "",
                "function": row.function,
                "level": level if isinstance(level, str) else "Not specified",
                "workType": row.formatted_work_type if isinstance(row.formatted_work_type, str) else "",
                "url": row.job_posting_url if isinstance(row.job_posting_url, str) else "",
                "description": truncate(row.description, DISPLAY_CHARS),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": {
            "name": "LinkedIn Job Postings (2023-2024)",
            "url": f"https://www.kaggle.com/datasets/{DATASET}",
            "license": "CC BY-SA 4.0",
            "totalPostings": total_postings,
            "sampling": f"up to {args.per_function} per job function, seed {args.seed}",
        },
        "jobs": jobs,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(jobs)} jobs across {sampled['function'].nunique()} functions to {OUT}")


if __name__ == "__main__":
    main()
