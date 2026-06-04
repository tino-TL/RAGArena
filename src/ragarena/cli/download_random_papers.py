from __future__ import annotations

import argparse
import asyncio
import hashlib
import random
import time
from pathlib import Path

from dotenv import load_dotenv
import requests

from ragarena.config import settings
from ragarena.papers.arxiv_client import fetch_arxiv_papers
from ragarena.papers.downloader import safe_arxiv_id
from ragarena.papers.models import PaperFile, PaperMetadata, StoredPaper
from ragarena.papers.repository import (
    ensure_papers_table,
    fetch_paper_by_arxiv_id,
    insert_paper_file,
    insert_papers,
)


DEFAULT_OUTPUT_DIR = Path("E:/ragarena-data/papers")
DEFAULT_CATEGORIES = [
    "cs.AI",
    "cs.CL",
    "cs.CV",
    "cs.LG",
    "cs.RO",
    "cs.SE",
    "cs.DB",
    "stat.ML",
    "math.OC",
    "math.PR",
    "econ.EM",
    "q-bio.NC",
    "physics.data-an",
    "astro-ph.IM",
]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Download a diverse random-ish arXiv PDF corpus for scale simulation"
    )
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--candidates-per-category", type=int, default=25)
    parser.add_argument("--max-start", type=int, default=300)
    parser.add_argument("--download-timeout", type=int, default=120)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--metadata-delay-seconds", type=float, default=3.0)
    args = parser.parse_args()

    asyncio.run(
        download_random_papers(
            target_count=args.target_count,
            output_dir=args.output_dir,
            seed=args.seed,
            categories=parse_csv(args.categories),
            candidates_per_category=args.candidates_per_category,
            max_start=args.max_start,
            download_timeout=args.download_timeout,
            delay_seconds=args.delay_seconds,
            metadata_delay_seconds=args.metadata_delay_seconds,
        )
    )


async def download_random_papers(
    *,
    target_count: int,
    output_dir: Path,
    seed: int,
    categories: list[str],
    candidates_per_category: int,
    max_start: int,
    download_timeout: int,
    delay_seconds: float,
    metadata_delay_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(seed)

    print(f"output_dir: {output_dir}")
    print(f"target_count: {target_count}")
    print(f"categories: {', '.join(categories)}")
    print(f"seed: {seed}")

    candidates = collect_candidates(
        categories=categories,
        candidates_per_category=candidates_per_category,
        max_start=max_start,
        randomizer=randomizer,
        metadata_delay_seconds=metadata_delay_seconds,
    )
    randomizer.shuffle(candidates)
    print(f"candidate_count: {len(candidates)}")

    await ensure_papers_table(settings.postgres_dsn)
    downloaded = 0
    skipped_existing = 0
    failed = 0
    seen_ids: set[str] = set()

    for metadata in candidates:
        if downloaded >= target_count:
            break
        if metadata.arxiv_id in seen_ids:
            continue
        seen_ids.add(metadata.arxiv_id)

        await insert_papers(settings.postgres_dsn, [metadata])
        paper = await fetch_paper_by_arxiv_id(settings.postgres_dsn, metadata.arxiv_id)
        if paper is None:
            failed += 1
            print(f"failed metadata lookup: {metadata.arxiv_id}")
            continue

        output_path = output_dir / f"{safe_arxiv_id(paper.arxiv_id)}.pdf"
        if output_path.exists() and output_path.stat().st_size > 0:
            skipped_existing += 1
            paper_file = paper_file_from_existing(paper, output_path)
            await insert_paper_file(settings.postgres_dsn, paper_file)
            downloaded += 1
            print(f"[{downloaded}/{target_count}] existing {paper.arxiv_id}: {output_path}")
            continue

        try:
            paper_file = download_pdf_streaming(
                paper,
                output_path=output_path,
                timeout=download_timeout,
            )
            await insert_paper_file(settings.postgres_dsn, paper_file)
            downloaded += 1
            size_mb = paper_file.file_size / 1024 / 1024
            print(f"[{downloaded}/{target_count}] downloaded {paper.arxiv_id} {size_mb:.2f} MB")
        except Exception as exc:
            failed += 1
            if output_path.exists() and output_path.stat().st_size == 0:
                output_path.unlink()
            print(f"failed {paper.arxiv_id}: {exc}")

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    print("download summary:")
    print(f"downloaded_or_existing: {downloaded}")
    print(f"skipped_existing: {skipped_existing}")
    print(f"failed: {failed}")
    print(f"output_dir: {output_dir}")


def collect_candidates(
    *,
    categories: list[str],
    candidates_per_category: int,
    max_start: int,
    randomizer: random.Random,
    metadata_delay_seconds: float,
) -> list[PaperMetadata]:
    by_id: dict[str, PaperMetadata] = {}
    for category in categories:
        start = randomizer.randint(0, max(0, max_start))
        query = f"cat:{category}"
        print(f"fetching metadata: query={query} start={start} max={candidates_per_category}")
        try:
            papers = fetch_arxiv_papers(
                query,
                max_results=candidates_per_category,
                start=start,
                timeout=60,
            )
        except Exception as exc:
            print(f"metadata fetch failed for {category}: {exc}")
            papers = []
        for paper in papers:
            by_id.setdefault(paper.arxiv_id, paper)
        if metadata_delay_seconds > 0:
            time.sleep(metadata_delay_seconds)
    return list(by_id.values())


def download_pdf_streaming(
    paper: StoredPaper,
    *,
    output_path: Path,
    timeout: int,
) -> PaperFile:
    hasher = hashlib.sha256()
    total_size = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    with requests.get(paper.pdf_url, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        with temporary_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                hasher.update(chunk)
                total_size += len(chunk)
    temporary_path.replace(output_path)
    return PaperFile(
        paper_id=paper.id,
        arxiv_id=paper.arxiv_id,
        pdf_url=paper.pdf_url,
        file_path=output_path,
        file_sha256=hasher.hexdigest(),
        file_size=total_size,
    )


def paper_file_from_existing(paper: StoredPaper, output_path: Path) -> PaperFile:
    hasher = hashlib.sha256()
    total_size = 0
    with output_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
            total_size += len(chunk)
    return PaperFile(
        paper_id=paper.id,
        arxiv_id=paper.arxiv_id,
        pdf_url=paper.pdf_url,
        file_path=output_path,
        file_sha256=hasher.hexdigest(),
        file_size=total_size,
    )


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
