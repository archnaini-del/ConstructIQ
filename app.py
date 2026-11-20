import hashlib
import io
import re
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

st.set_page_config(
    page_title="ConstructIQ Tender Intelligence",
    page_icon="🏗️",
    layout="wide",
)

st.title("🏗️ ConstructIQ Tender Intelligence")
st.caption("Upload tender PDFs and extract their text for review.")


def clean_filename(filename: str, index: int) -> str:
    name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[\x00-\x1f\x7f]+", "", name).strip()
    return name or f"document_{index}.pdf"


def safe_txt_stem(filename: str, index: int) -> str:
    stem = Path(clean_filename(filename, index)).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem[:80] or f"document_{index}"


def extract_pdf_text(pdf_bytes: bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))

    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise ValueError("The PDF is encrypted and cannot be opened.") from exc
        if not unlocked:
            raise ValueError("The PDF requires a password.")

    pages = []
    failed_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append(f"--- Page {page_number} ---\n{page_text}")
        except Exception:
            failed_pages.append(page_number)

    return "\n\n".join(pages).strip(), len(reader.pages), failed_pages


uploaded_files = st.file_uploader(
    "Upload tender documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more text-based PDF files. Scanned PDFs require OCR, which is not included yet.",
)

if not uploaded_files:
    st.info("Upload at least one PDF to begin.")
    st.stop()

st.info(f"{len(uploaded_files)} PDF file(s) selected.")

if st.button("Extract PDF text", type="primary"):
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        display_name = clean_filename(uploaded_file.name, index)
        pdf_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()[:12]

        with st.expander(display_name, expanded=len(uploaded_files) == 1):
            try:
                text, page_count, failed_pages = extract_pdf_text(pdf_bytes)
            except Exception as exc:
                st.error(f"Could not read this PDF: {exc}")
                continue

            if not text:
                st.warning(
                    "No extractable text was found. The PDF may be scanned or image-only. "
                    "OCR is not included in this MVP."
                )
                continue

            st.success(f"Text extracted from {page_count} page(s).")

            if failed_pages:
                page_list = ", ".join(map(str, failed_pages))
                st.warning(f"Text could not be extracted from page(s): {page_list}")

            st.text_area(
                "Extracted text",
                value=text,
                height=500,
                disabled=True,
                key=f"text_{index}_{file_hash}",
            )

            st.download_button(
                "Download extracted text (.txt)",
                data=text,
                file_name=f"{safe_txt_stem(uploaded_file.name, index)}-extracted.txt",
                mime="text/plain",
                key=f"download_{index}_{file_hash}",
            )

