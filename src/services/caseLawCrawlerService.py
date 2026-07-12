from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from bs4 import BeautifulSoup
from pypdf import PdfReader

from src.config.index import appConfig
from src.services.awsS3 import s3_client
from src.services.supabase import supabase
from src.services.webScrapper import scrapingbee_client

ANLE_BASE = "https://anle.toaan.gov.vn"
ANLE_LIST_PATH = "/webcenter/portal/anle/anle"
ANLE_DETAIL_PATH = "/webcenter/portal/anle/chitietanle"
ANLE_PDF_PATH = "/webcenter/ShowProperty"

CASE_NUMBER_RE = re.compile(r"^(\d+/\d+/AL)$", re.IGNORECASE)
DETAIL_RE = re.compile(r"chitietanle\?dDocName=(TAND\d+)", re.IGNORECASE)
SCRAPINGBEE_HEX_ESCAPE_RE = re.compile(rb"\\x([0-9A-Fa-f]{2})")

DEFAULT_LINH_VUC = 34  # Dân sự
DEFAULT_MUC_HIEN_THI = 9009
REQUEST_PAUSE_SECONDS = 1.2
TESSDATA_DIR = Path(__file__).resolve().parents[2] / "tessdata"
TESSERACT_CANDIDATES = [
    os.environ.get("TESSERACT_CMD"),
    shutil.which("tesseract"),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
]

LINH_VUC_LABELS = {
    34: "Dân sự",
}


def _scrapingbee_get(
    url: str,
    *,
    render_js: bool = True,
    wait_ms: int = 5000,
) -> bytes:
    response = scrapingbee_client.get(
        url,
        params={
            "render_js": "true" if render_js else "false",
            "wait": wait_ms if render_js else 0,
            "premium_proxy": "true",
            "country_code": "vn",
        },
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"ScrapingBee failed ({response.status_code}) for {url}: "
            f"{response.content[:300]!r}"
        )
    return response.content


def build_listing_url(
    *,
    linh_vuc: int = DEFAULT_LINH_VUC,
    page: int = 1,
    muc_hien_thi: int = DEFAULT_MUC_HIEN_THI,
) -> str:
    if page <= 1:
        return f"{ANLE_BASE}{ANLE_LIST_PATH}?linhVuc={linh_vuc}"
    return (
        f"{ANLE_BASE}{ANLE_LIST_PATH}"
        f"?selectedPage={page}&docType=AnLe&mucHienThi={muc_hien_thi}&linhVuc={linh_vuc}"
    )


def build_detail_url(d_doc_name: str) -> str:
    return f"{ANLE_BASE}{ANLE_DETAIL_PATH}?dDocName={d_doc_name}"


def build_pdf_url(d_doc_name: str) -> str:
    return f"{ANLE_BASE}{ANLE_PDF_PATH}?nodeId=/UCMServer/{d_doc_name}"


def parse_attributes_text(attributes_text: str | None) -> dict[str, Optional[str]]:
    text = attributes_text or ""
    patterns = {
        "adopted_date": r"Ngày thông qua\s+(\d{2}/\d{2}/\d{4})",
        "published_date": r"Ngày công bố\s+(\d{2}/\d{2}/\d{4})",
        "effective_date": r"Ngày áp dụng\s+(\d{2}/\d{2}/\d{4})",
        "status": r"Trạng thái\s+(.+?)(?:\s+Ngày|\s*$)",
        "linh_vuc_label": r"Lĩnh vực\s+(.+?)(?:\s+Ngày|\s+Trạng thái|\s*$)",
        "case_number": r"Số án lệ\s+(\d+/\d+/AL)",
    }
    parsed: dict[str, Optional[str]] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        parsed[key] = match.group(1).strip() if match else None
    return parsed


def repair_scrapingbee_pdf_bytes(pdf_bytes: bytes) -> bytes:
    """
    ScrapingBee can escape binary PDF streams as ASCII sequences like \\xFF.
    Rebuild real bytes so embedded JPEG images remain readable.
    """
    if pdf_bytes.count(b"\\x") < 50:
        return pdf_bytes
    repaired = SCRAPINGBEE_HEX_ESCAPE_RE.sub(
        lambda match: bytes([int(match.group(1), 16)]),
        pdf_bytes,
    )
    return repaired if repaired.startswith(b"%PDF") else pdf_bytes


def _resolve_tesseract_cmd() -> str | None:
    for candidate in TESSERACT_CANDIDATES:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _ensure_tessdata() -> Path:
    TESSDATA_DIR.mkdir(parents=True, exist_ok=True)
    vie_path = TESSDATA_DIR / "vie.traineddata"
    if vie_path.exists():
        return TESSDATA_DIR

    # Prefer copying from a local Tesseract install when available.
    system_vie = Path(r"C:\Program Files\Tesseract-OCR\tessdata\vie.traineddata")
    if system_vie.exists():
        shutil.copy2(system_vie, vie_path)
        return TESSDATA_DIR

    import urllib.request

    urllib.request.urlretrieve(
        "https://github.com/tesseract-ocr/tessdata/raw/main/vie.traineddata",
        vie_path,
    )
    return TESSDATA_DIR


def _ocr_image_bytes(image_bytes: bytes) -> str:
    tesseract_cmd = _resolve_tesseract_cmd()
    if not tesseract_cmd:
        raise RuntimeError(
            "Không tìm thấy Tesseract OCR. Cài Tesseract và/hoặc set TESSERACT_CMD."
        )

    tessdata_dir = _ensure_tessdata()
    # Prefer Vietnamese; fall back to eng if vie model is unavailable at runtime.
    languages = "vie"
    if not (tessdata_dir / "vie.traineddata").exists():
        languages = "eng"

    completed = subprocess.run(
        [
            tesseract_cmd,
            "stdin",
            "stdout",
            "-l",
            languages,
            "--tessdata-dir",
            str(tessdata_dir),
            "--psm",
            "6",
        ],
        input=image_bytes,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(stderr or "Tesseract OCR failed")
    return completed.stdout.decode("utf-8", errors="ignore").strip()


def extract_pdf_text_layer(pdf_bytes: bytes, max_chars: int = 500_000) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
    parts: list[str] = []
    total = 0
    for page in reader.pages:
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue
        parts.append(page_text)
        total += len(page_text)
        if total >= max_chars:
            break
    text = "\n\n".join(parts).strip()
    return text[:max_chars] if len(text) > max_chars else text


def extract_pdf_text_via_ocr(pdf_bytes: bytes, max_chars: int = 500_000) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
    parts: list[str] = []
    total = 0

    for page_index, page in enumerate(reader.pages):
        resources = page.get("/Resources")
        if not resources:
            continue
        xobjects = resources.get("/XObject")
        if not xobjects:
            continue

        for name in xobjects:
            try:
                image_obj = xobjects[name].get_object()
            except Exception:
                continue
            if image_obj.get("/Subtype") != "/Image":
                continue

            filters = image_obj.get("/Filter")
            filter_names = (
                [str(item) for item in filters]
                if isinstance(filters, list)
                else [str(filters)]
            )
            if "/DCTDecode" not in filter_names and "DCTDecode" not in "".join(
                filter_names
            ):
                continue

            try:
                image_bytes = image_obj.get_data()
            except Exception:
                continue
            if not image_bytes.startswith(b"\xff\xd8"):
                continue

            page_text = _ocr_image_bytes(image_bytes)
            if not page_text:
                continue
            parts.append(page_text)
            total += len(page_text)
            if total >= max_chars:
                break
        if total >= max_chars:
            break

        # Soft rate-limit OCR load on large scanned PDFs.
        if page_index < len(reader.pages) - 1:
            time.sleep(0.05)

    text = "\n\n".join(parts).strip()
    return text[:max_chars] if len(text) > max_chars else text


def extract_pdf_text(pdf_bytes: bytes, max_chars: int = 500_000) -> str:
    text = extract_pdf_text_layer(pdf_bytes, max_chars=max_chars)
    if len(text.strip()) >= 80:
        return text
    return extract_pdf_text_via_ocr(pdf_bytes, max_chars=max_chars)


def parse_listing_html(html: str | bytes) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    cases: dict[str, dict[str, Any]] = {}

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        match = DETAIL_RE.search(href)
        if not match:
            continue
        if re.search(r"(?:^|[?&])Tab=", href, re.IGNORECASE):
            continue

        d_doc_name = match.group(1).upper()
        text = " ".join(anchor.get_text(" ", strip=True).split())
        if not text:
            continue

        case = cases.setdefault(
            d_doc_name,
            {
                "dDocName": d_doc_name,
                "caseNumber": None,
                "title": None,
                "detailUrl": build_detail_url(d_doc_name),
                "attributesText": None,
            },
        )

        if CASE_NUMBER_RE.fullmatch(text):
            case["caseNumber"] = text.upper()
            continue

        if text.startswith("Thuộc tính"):
            case["attributesText"] = text
            continue

        if text in {"Văn bản liên quan", "Tải về", "In"}:
            continue

        if len(text) >= 12 and (
            not case["title"] or len(text) > len(case["title"] or "")
        ):
            case["title"] = text

    results: list[dict[str, Any]] = []
    for case in cases.values():
        if not case["caseNumber"] and not case["title"]:
            continue
        if not case["title"]:
            case["title"] = f"Án lệ {case['caseNumber'] or case['dDocName']}"
        if not case["caseNumber"]:
            case["caseNumber"] = case["dDocName"]
        results.append(case)

    results.sort(key=lambda item: item["caseNumber"], reverse=True)
    return results


def discover_max_page(html: str | bytes) -> int:
    soup = BeautifulSoup(html, "html.parser")
    max_page = 1
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        page_match = re.search(r"selectedPage=(\d+)", href)
        if page_match:
            max_page = max(max_page, int(page_match.group(1)))
        text = anchor.get_text(strip=True)
        if text.isdigit():
            max_page = max(max_page, int(text))
    return max_page


def discover_case_laws(
    *,
    linh_vuc: int = DEFAULT_LINH_VUC,
    max_pages: int = 5,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1")

    discovered: dict[str, dict[str, Any]] = {}
    first_html = _scrapingbee_get(build_listing_url(linh_vuc=linh_vuc, page=1))
    available_pages = min(max_pages, discover_max_page(first_html))

    for page in range(1, available_pages + 1):
        html = (
            first_html
            if page == 1
            else _scrapingbee_get(build_listing_url(linh_vuc=linh_vuc, page=page))
        )
        for case in parse_listing_html(html):
            discovered[case["dDocName"]] = case
            if max_items is not None and len(discovered) >= max_items:
                return list(discovered.values())[:max_items]
        if page < available_pages:
            time.sleep(REQUEST_PAUSE_SECONDS)

    return list(discovered.values())


def download_case_pdf(d_doc_name: str) -> bytes:
    content = _scrapingbee_get(build_pdf_url(d_doc_name), render_js=False, wait_ms=0)
    if not content.startswith(b"%PDF"):
        raise RuntimeError(
            f"Expected PDF for {d_doc_name}, got magic={content[:16]!r}"
        )
    return repair_scrapingbee_pdf_bytes(content)


def find_existing_case_law(d_doc_name: str) -> dict | None:
    result = (
        supabase.table("case_laws")
        .select("id, case_number, title, processing_status, source_url")
        .eq("d_doc_name", d_doc_name)
        .limit(1)
        .execute()
    )
    if result.data:
        return result.data[0]
    return None


def ingest_case_law(case: dict[str, Any]) -> dict[str, Any]:
    d_doc_name = case["dDocName"]
    detail_url = case.get("detailUrl") or build_detail_url(d_doc_name)
    existing = find_existing_case_law(d_doc_name)
    if existing and existing.get("processing_status") == "completed":
        return {
            "status": "skipped",
            "reason": "already_exists",
            "caseLawId": existing["id"],
            "caseNumber": case.get("caseNumber") or existing.get("case_number"),
            "title": case.get("title") or existing.get("title"),
            "detailUrl": detail_url,
        }

    attributes = parse_attributes_text(case.get("attributesText"))
    linh_vuc = case.get("linhVuc")
    linh_vuc_label = (
        attributes.get("linh_vuc_label")
        or LINH_VUC_LABELS.get(linh_vuc)
        or None
    )
    case_number = case.get("caseNumber") or attributes.get("case_number")

    pdf_bytes = download_case_pdf(d_doc_name)
    s3_key = f"case-laws/{d_doc_name}-{uuid.uuid4().hex[:8]}.pdf"
    s3_client.put_object(
        Bucket=appConfig["s3_bucket_name"],
        Key=s3_key,
        Body=pdf_bytes,
        ContentType="application/pdf",
    )

    full_text = ""
    extract_error = None
    try:
        full_text = extract_pdf_text(pdf_bytes)
    except Exception as error:
        extract_error = str(error)

    # Still keep the record searchable by số/tiêu đề even if OCR fails.
    processing_status = (
        "completed" if (full_text or case_number or case.get("title")) else "failed"
    )
    row = {
        "d_doc_name": d_doc_name,
        "case_number": case_number,
        "title": case.get("title") or f"Án lệ {case_number or d_doc_name}",
        "linh_vuc": linh_vuc,
        "linh_vuc_label": linh_vuc_label,
        "adopted_date": attributes.get("adopted_date"),
        "published_date": attributes.get("published_date"),
        "effective_date": attributes.get("effective_date"),
        "status": attributes.get("status"),
        "source_url": detail_url,
        "pdf_url": build_pdf_url(d_doc_name),
        "s3_key": s3_key,
        "file_size": len(pdf_bytes),
        "full_text": full_text,
        "attributes_text": case.get("attributesText"),
        "metadata": {
            "source": "anle.toaan.gov.vn",
            "docType": "AnLe",
            "textExtraction": "ocr" if full_text else "failed",
        },
        "processing_status": processing_status,
        "error_message": extract_error,
    }

    if existing:
        update_result = (
            supabase.table("case_laws")
            .update(row)
            .eq("id", existing["id"])
            .execute()
        )
        if not update_result.data:
            raise RuntimeError(f"Failed to update case_laws row for {d_doc_name}")
        created = update_result.data[0]
    else:
        insert_result = supabase.table("case_laws").insert(row).execute()
        if not insert_result.data:
            raise RuntimeError(f"Failed to create case_laws row for {d_doc_name}")
        created = insert_result.data[0]

    return {
        "status": "saved" if processing_status == "completed" else "failed",
        "caseLawId": created["id"],
        "caseNumber": created.get("case_number"),
        "title": created.get("title"),
        "detailUrl": detail_url,
        "processingStatus": processing_status,
        "error": extract_error,
    }


def crawl_and_ingest_case_laws(
    *,
    linh_vuc: int = DEFAULT_LINH_VUC,
    max_pages: int = 5,
    max_items: int | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    from src.services.caseLawCrawlJobStore import (
        append_recent_item,
        update_crawl_job,
    )

    def report(**fields: Any) -> None:
        if job_id:
            update_crawl_job(job_id, **fields)

    report(
        status="running",
        percent=5,
        message="Đang thu thập danh sách án lệ...",
        error=None,
    )

    cases = discover_case_laws(
        linh_vuc=linh_vuc,
        max_pages=max_pages,
        max_items=max_items,
    )
    for case in cases:
        case["linhVuc"] = linh_vuc

    total = len(cases)
    report(
        status="running",
        percent=15 if total else 100,
        message=(
            f"Đã tìm thấy {total} án lệ. Đang lưu vào kho tra cứu..."
            if total
            else "Không tìm thấy án lệ nào trên các trang đã quét."
        ),
        discovered=total,
        processed=0,
        queued=0,
        skipped=0,
        failed=0,
    )

    queued: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for index, case in enumerate(cases):
        case_label = case.get("caseNumber") or case.get("dDocName")
        report(
            status="running",
            currentCase={
                "caseNumber": case.get("caseNumber"),
                "title": case.get("title"),
                "detailUrl": case.get("detailUrl"),
            },
            message=f"Đang lưu {case_label} vào kho tra cứu ({index + 1}/{total})...",
            percent=15 + int((index / max(total, 1)) * 80),
            processed=index,
            queued=len(queued),
            skipped=len(skipped),
            failed=len(failed),
        )

        try:
            result = ingest_case_law(case)
            if result["status"] == "skipped":
                skipped.append(result)
            elif result["status"] == "failed":
                failed.append(result)
            else:
                queued.append(result)
            if job_id:
                append_recent_item(job_id, result)
        except Exception as error:
            failed_item = {
                "status": "failed",
                "caseNumber": case.get("caseNumber"),
                "title": case.get("title"),
                "detailUrl": case.get("detailUrl"),
                "error": str(error),
            }
            failed.append(failed_item)
            if job_id:
                append_recent_item(job_id, failed_item)

        if index < len(cases) - 1:
            time.sleep(REQUEST_PAUSE_SECONDS)

    summary = {
        "linhVuc": linh_vuc,
        "discovered": total,
        "queued": len(queued),
        "skipped": len(skipped),
        "failed": len(failed),
        "items": {
            "queued": queued,
            "skipped": skipped,
            "failed": failed,
        },
    }

    report(
        status="completed",
        percent=100,
        message=(
            f"Hoàn tất: {len(queued)} đã lưu kho tra cứu, "
            f"{len(skipped)} bỏ qua, {len(failed)} lỗi."
        ),
        discovered=total,
        processed=total,
        queued=len(queued),
        skipped=len(skipped),
        failed=len(failed),
        currentCase=None,
        error=None,
    )
    return summary
