#!/usr/bin/env python
"""
Fetch open-access (OA) PDFs for all references of a given paper.

This script is designed to be *copyright-safe*:
  - It only downloads PDFs that are publicly accessible via OA links.
  - It does NOT attempt to bypass paywalls.

Typical use (Windows / PowerShell):
  python simveri_validation/scripts/paper_fetch_reference_pdfs.py `
    --paper-pdf "./paper.pdf" `
    --out-dir  "./reference_pdfs"

Outputs (under a new subfolder in --out-dir):
  - manifest.csv                 (one row per reference)
  - source_work_openalex.json    (OpenAlex metadata for the source paper)
  - references_openalex.jsonl    (one JSON per reference, as returned by OpenAlex)
  - pdfs/                        (downloaded PDFs, OA only)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests


DOI_RE = re.compile(r"10\.[0-9]{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch OA PDFs for a paper's references (OpenAlex-based).")
    p.add_argument("--paper-pdf", type=str, required=True, help="Local PDF path of the source paper.")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory (a subfolder will be created).")
    p.add_argument(
        "--mailto",
        type=str,
        default="",
        help="(Recommended) Contact email for OpenAlex polite usage, e.g. name@domain.com",
    )
    p.add_argument("--max-refs", type=int, default=0, help="Limit number of references to process (0=all).")
    p.add_argument("--max-mb", type=float, default=80.0, help="Max single PDF size (MB).")
    p.add_argument("--timeout-s", type=float, default=30.0, help="HTTP timeout seconds.")
    p.add_argument("--sleep-s", type=float, default=0.05, help="Sleep between OpenAlex requests (seconds).")
    p.add_argument("--dry-run", action="store_true", help="Only build manifest; do not download PDFs.")
    return p.parse_args()


def _safe_slug(s: str, max_len: int = 120) -> str:
    s = (s or "").strip()
    if not s:
        return "unknown"
    s = s.replace("https://doi.org/", "").replace("http://doi.org/", "")
    s = s.replace("https://openalex.org/", "").replace("http://openalex.org/", "")
    s = re.sub(r"[\\/:*groups\"<>|\r\n\t]+", "_", s)
    s = re.sub(r"\s+", "_", s)
    return s[:max_len].strip("_") or "unknown"


def _extract_dois_from_pdf(pdf_path: Path, max_pages: int = 3) -> List[str]:
    try:
        import fitz  # PyMuPDF
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyMuPDF (fitz) is required but not available") from e

    doc = fitz.open(str(pdf_path))
    try:
        text = ""
        for i in range(min(max_pages, doc.page_count)):
            text += doc.load_page(i).get_text("text") + "\n"
    finally:
        doc.close()

    dois = sorted({m.group(0).rstrip(").,;") for m in DOI_RE.finditer(text)})
    return dois


def _pick_source_doi(dois: Sequence[str]) -> str:
    if not dois:
        raise ValueError("No DOI found in the first pages of the source PDF.")
    # Prefer the longest DOI string in case of multiple matches.
    return sorted(dois, key=len, reverse=True)[0]


def _openalex_get(session: requests.Session, url: str, timeout_s: float) -> dict:
    r = session.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _openalex_work_url_for_doi(doi: str, mailto: str) -> str:
    base = f"https://api.openalex.org/works/https://doi.org/{doi}"
    if mailto:
        return f"{base}groupsmailto={requests.utils.quote(mailto)}"
    return base


def _openalex_batch_works_url(openalex_ids: Sequence[str], mailto: str) -> str:
    # OpenAlex supports OR via `filter=openalex_id:W1|W2|...`.
    # Note: URL length can explode if the batch is too large; we chunk in caller.
    filt = "openalex_id:" + "|".join(openalex_ids)
    base = f"https://api.openalex.org/worksgroupsfilter={requests.utils.quote(filt, safe=':|')}&per-page=200"
    if mailto:
        return f"{base}&mailto={requests.utils.quote(mailto)}"
    return base


def _chunk_by_url_len(ids: Sequence[str], max_url_len: int = 7600) -> List[List[str]]:
    # Conservative URL length cap for Windows / proxies.
    batches: List[List[str]] = []
    cur: List[str] = []
    cur_len = 0
    for oid in ids:
        add = len(oid) + (1 if cur else 0)
        if cur and (cur_len + add) > max_url_len:
            batches.append(cur)
            cur = [oid]
            cur_len = len(oid)
        else:
            cur.append(oid)
            cur_len += add
    if cur:
        batches.append(cur)
    return batches


def _chunk_ids(ids: Sequence[str], batch_size: int) -> List[List[str]]:
    if batch_size <= 0:
        return [list(ids)]
    return [list(ids[i : i + batch_size]) for i in range(0, len(ids), batch_size)]


@dataclass
class RefRecord:
    openalex_id: str
    doi: str
    title: str
    pdf_url: str
    oa_source: str


def _extract_oa_pdf(rec: dict) -> Tuple[str, str]:
    """
    Return (pdf_url, source_label). Empty pdf_url means "no direct OA PDF".
    """
    def _get(d: dict, *keys: str) -> str:
        cur = d
        for k in keys:
            if not isinstance(cur, dict):
                return ""
            cur = cur.get(k)
        return str(cur) if isinstance(cur, str) else ""

    # 1) Best OA location direct PDF
    pdf = _get(rec, "best_oa_location", "pdf_url")
    if pdf:
        return pdf, "openalex.best_oa_location.pdf_url"

    # 2) Primary location direct PDF
    pdf = _get(rec, "primary_location", "pdf_url")
    if pdf:
        return pdf, "openalex.primary_location.pdf_url"

    # 3) OA URL sometimes is a direct PDF
    oa_url = _get(rec, "open_access", "oa_url")
    if oa_url.lower().endswith(".pdf"):
        return oa_url, "openalex.open_access.oa_url"

    # 4) arXiv landing -> PDF
    for cand in (
        _get(rec, "best_oa_location", "landing_page_url"),
        _get(rec, "primary_location", "landing_page_url"),
        oa_url,
    ):
        if "arxiv.org/abs/" in (cand or ""):
            arxiv_id = cand.split("arxiv.org/abs/", 1)[1].split("groups", 1)[0].strip("/")
            if arxiv_id:
                return f"https://arxiv.org/pdf/{arxiv_id}.pdf", "arxiv.abs_to_pdf"

    return "", ""


def _download_pdf(
    session: requests.Session,
    url: str,
    out_path: Path,
    timeout_s: float,
    max_mb: float,
) -> Tuple[bool, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with session.get(url, timeout=timeout_s, stream=True, allow_redirects=True) as r:
            r.raise_for_status()

            ctype = str(r.headers.get("Content-Type", "")).lower()
            # Some servers use octet-stream for PDFs; allow that.
            if ("pdf" not in ctype) and ("octet-stream" not in ctype):
                # Heuristic: allow unknown content-type, but verify PDF magic.
                pass

            clen = r.headers.get("Content-Length")
            if clen is not None:
                try:
                    size = int(clen)
                    if size > int(max_mb * 1024 * 1024):
                        return False, f"too_large(Content-Length={size} bytes)"
                except Exception:
                    pass

            max_bytes = int(max_mb * 1024 * 1024)
            seen = 0
            first = b""
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if not chunk:
                        continue
                    if not first:
                        first = chunk[:8]
                    f.write(chunk)
                    seen += len(chunk)
                    if seen > max_bytes:
                        try:
                            tmp_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        return False, f"too_large(streamed>{max_mb}MB)"

            # Verify PDF magic
            if not first.startswith(b"%PDF"):
                # Not a PDF (likely HTML). Remove.
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return False, "not_a_pdf(magic_mismatch)"

            tmp_path.replace(out_path)
            return True, "ok"
    except Exception as e:
        return False, f"download_error({type(e).__name__})"


def main() -> None:
    args = parse_args()

    paper_pdf = Path(args.paper_pdf).expanduser().resolve()
    if not paper_pdf.exists():
        raise FileNotFoundError(str(paper_pdf))
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dois = _extract_dois_from_pdf(paper_pdf, max_pages=3)
    source_doi = _pick_source_doi(dois)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": f"paper_fetch_reference_pdfs/1.0 (mailto={args.mailto or 'unknown'})",
            "Accept": "application/json",
        }
    )

    # Resolve the source paper in OpenAlex
    src_url = _openalex_work_url_for_doi(source_doi, args.mailto)
    src = _openalex_get(session, src_url, timeout_s=float(args.timeout_s))

    src_title = str(src.get("title") or src.get("display_name") or "")
    src_year = src.get("publication_year")
    src_slug = _safe_slug(paper_pdf.stem)
    if source_doi:
        src_slug = f"{src_slug}__{_safe_slug(source_doi)}"
    run_root = out_dir / src_slug
    pdf_dir = run_root / "pdfs"
    run_root.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Save source metadata
    (run_root / "source_work_openalex.json").write_text(json.dumps(src, ensure_ascii=False, indent=2), encoding="utf-8")

    referenced = list(src.get("referenced_works") or [])
    openalex_ids = [str(x).split("/")[-1] for x in referenced if isinstance(x, str) and x]
    if args.max_refs and args.max_refs > 0:
        openalex_ids = openalex_ids[: int(args.max_refs)]

    # Fetch references metadata in batches
    rec_by_openalex_id: Dict[str, dict] = {}

    # OpenAlex may reject overly long OR-filters; keep batches small and URL-safe.
    batches: List[List[str]] = []
    for b in _chunk_ids(openalex_ids, batch_size=50):
        batches.extend(_chunk_by_url_len(b, max_url_len=7600))

    def fetch_openalex_batch(batch_ids: List[str]) -> List[dict]:
        url = _openalex_batch_works_url(batch_ids, args.mailto)
        try:
            data = _openalex_get(session, url, timeout_s=float(args.timeout_s))
        except requests.HTTPError as e:
            # If OpenAlex rejects the batch (e.g., 400), split and retry.
            if len(batch_ids) <= 1:
                raise
            if getattr(e.response, "status_code", None) == 400:
                mid = len(batch_ids) // 2
                left = fetch_openalex_batch(batch_ids[:mid])
                right = fetch_openalex_batch(batch_ids[mid:])
                return left + right
            raise
        results = data.get("results") or []
        return results if isinstance(results, list) else []

    jsonl_path = run_root / "references_openalex.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for batch in batches:
            batch_ids = list(dict.fromkeys(batch))

            results = fetch_openalex_batch(batch_ids)

            for rec in results:
                if not isinstance(rec, dict):
                    continue
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                oa_pdf, oa_src = _extract_oa_pdf(rec)
                oid = str(rec.get("id") or "")
                oid_short = oid.split("/")[-1] if oid else ""
                if oid_short:
                    rec_by_openalex_id[oid_short] = rec

            time.sleep(float(args.sleep_s))

    # Build a stable reference list in the original order. For OpenAlex-missing IDs, create placeholders so
    # the manifest rows match `len(referenced_works)`.
    ref_recs: List[RefRecord] = []
    missing_openalex = 0
    for oid in openalex_ids:
        rec = rec_by_openalex_id.get(oid)
        if not isinstance(rec, dict):
            missing_openalex += 1
            ref_recs.append(
                RefRecord(
                    openalex_id=str(oid),
                    doi="",
                    title="",
                    pdf_url="",
                    oa_source="",
                )
            )
            continue

        oa_pdf, oa_src = _extract_oa_pdf(rec)
        doi = str(rec.get("doi") or "")
        doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        title = str(rec.get("title") or rec.get("display_name") or "")
        ref_recs.append(
            RefRecord(
                openalex_id=str(oid),
                doi=doi,
                title=title,
                pdf_url=oa_pdf,
                oa_source=oa_src,
            )
        )

    # Persist missing OpenAlex IDs for transparency/debugging.
    if missing_openalex:
        miss_path = run_root / "missing_openalex_ids.txt"
        missing = [oid for oid in openalex_ids if oid not in rec_by_openalex_id]
        miss_path.write_text("\n".join(missing) + "\n", encoding="utf-8")

    # Write manifest + download PDFs
    manifest_path = run_root / "manifest.csv"
    downloaded = 0
    available = 0

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "openalex_id",
                "doi",
                "title",
                "oa_pdf_url",
                "oa_source",
                "downloaded",
                "pdf_path",
                "note",
            ],
        )
        w.writeheader()

        for rr in ref_recs:
            row: Dict[str, str] = {
                "openalex_id": rr.openalex_id,
                "doi": rr.doi,
                "title": rr.title,
                "oa_pdf_url": rr.pdf_url,
                "oa_source": rr.oa_source,
                "downloaded": "0",
                "pdf_path": "",
                "note": "",
            }

            if not rr.pdf_url:
                row["note"] = "no_direct_oa_pdf_url"
                w.writerow(row)
                continue

            available += 1

            # Decide filename
            if rr.doi:
                fname = f"doi__{_safe_slug(rr.doi, max_len=180)}.pdf"
            elif rr.openalex_id:
                fname = f"openalex__{_safe_slug(rr.openalex_id)}.pdf"
            else:
                fname = f"ref__{_safe_slug(rr.title)}.pdf"

            out_pdf = pdf_dir / fname
            row["pdf_path"] = str(out_pdf)

            if args.dry_run:
                row["note"] = "dry_run"
                w.writerow(row)
                continue

            ok, note = _download_pdf(
                session=session,
                url=rr.pdf_url,
                out_path=out_pdf,
                timeout_s=float(args.timeout_s),
                max_mb=float(args.max_mb),
            )
            row["downloaded"] = "1" if ok else "0"
            row["note"] = note
            if ok:
                downloaded += 1
            w.writerow(row)

    # Friendly console summary (ASCII-safe)
    print("=" * 70)
    print("Reference OA PDF Fetch - Completed")
    print("=" * 70)
    print(f"Source DOI:     {source_doi}")
    print(f"Source title:   {src_title}")
    print(f"Source year:    {src_year}")
    print(f"Total refs:     {len(openalex_ids)}")
    print(f"OA pdf URLs:    {available}")
    print(f"Downloaded:     {downloaded}")
    print(f"Output folder:  {run_root}")
    print(f"Manifest:       {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
