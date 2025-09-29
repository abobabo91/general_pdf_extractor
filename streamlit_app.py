import streamlit as st
st.set_page_config(layout="centered")  # not wide
import pandas as pd
import PyPDF2
from pdf2image import convert_from_bytes
import pytesseract
from io import BytesIO
from PIL import Image, ImageSequence
from openai import OpenAI
import traceback, sys, re, gc
from docx import Document
from bs4 import BeautifulSoup
import json  # this one stays, but no pip install needed


# ---------------------------
# Global exception handler
# ---------------------------
def global_exception_handler(exc_type, exc_value, exc_traceback):
    st.error("Oops! Something went wrong:")
    st.code("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
sys.excepthook = global_exception_handler

# ---------------------------
# Session State
# ---------------------------
if "custom_fields" not in st.session_state:
    st.session_state.custom_fields = ["", "", ""]  # 3 inputs by default
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = []
if "df_extracted" not in st.session_state:
    st.session_state.df_extracted = pd.DataFrame()
if "final_prompt" not in st.session_state:
    st.session_state.final_prompt = ""

# ---------------------------
# OpenAI setup (fixed model)
# ---------------------------
client = OpenAI(api_key=st.secrets["openai"]["OPENAI_API_KEY"])
OPENAI_MODEL = "gpt-4o"  # do not change

# ---------------------------
# Text extraction (PDFs + Images)
# ---------------------------

def extract_text_from_upload(uploaded_file):
    """
    Extracts text from a single uploaded file.
    - PDFs: try native text; if sparse, OCR each page.
    - Images: OCR directly (supports multi-frame TIFF).
    """
    file_name = uploaded_file.name.lower()
    content_type = uploaded_file.type or ""

    # Helper: OCR a PIL image/frame
    def ocr_pil_image(img):
        return pytesseract.image_to_string(img, lang="eng+hun") or ""

    # --- PDF branch ---
    if file_name.endswith(".pdf") or "pdf" in content_type:
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF {uploaded_file.name}: {e}")
            return None

        # OCR fallback if too little native text
        if len(text.strip()) < 100:
            try:
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                num_pages = len(PyPDF2.PdfReader(BytesIO(file_bytes)).pages)
                per_pdf_progress = st.progress(0, text=f"OCR on {uploaded_file.name}...")
                text = ""
                for i in range(1, num_pages + 1):
                    images = convert_from_bytes(file_bytes, dpi=150, first_page=i, last_page=i)
                    text += ocr_pil_image(images[0]) + "\n"
                    del images
                    gc.collect()
                    per_pdf_progress.progress(i / num_pages, text=f"OCR on {uploaded_file.name} ({i}/{num_pages})")
                per_pdf_progress.empty()
            except Exception as e:
                st.error(f"OCR error for PDF {uploaded_file.name}: {e}")
                return None

        if len(text) > 300000:
            st.warning(f"{uploaded_file.name} is very long; only the first 300,000 characters will be processed.")
            text = text[:300000]
        return text

    # --- Image branch ---
    elif content_type.startswith("image/") or file_name.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")):
        try:
            img = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Error opening image {uploaded_file.name}: {e}")
            return None

        # Handle multi-frame images (e.g., TIFF)
        frames = []
        try:
            for i, frame in enumerate(ImageSequence.Iterator(img)):
                frames.append(frame.copy())
        except Exception:
            frames = [img]

        text = ""
        per_img_progress = st.progress(0, text=f"OCR on {uploaded_file.name}...")
        total = len(frames)
        for idx, frame in enumerate(frames, start=1):
            text += pytesseract.image_to_string(frame, lang="eng+hun") + "\n"
            per_img_progress.progress(idx / total, text=f"OCR on {uploaded_file.name} ({idx}/{total})")
        per_img_progress.empty()

        if len(text) > 300000:
            st.warning(f"{uploaded_file.name} is very long; only the first 300,000 characters will be processed.")
            text = text[:300000]
        return text

    else:
        st.warning(f"Unsupported file type: {uploaded_file.name}")
        return None
    
    # --- HTML branch ---
    elif file_name.endswith(".html") or "html" in content_type:
        try:
            soup = BeautifulSoup(uploaded_file.read(), "html.parser")
            text = soup.get_text(separator="\n")
            return text[:300000] if len(text) > 300000 else text
        except Exception as e:
            st.error(f"Error reading HTML {uploaded_file.name}: {e}")
            return None

    # --- DOCX branch ---
    elif file_name.endswith(".docx") or "word" in content_type:
        try:
            doc = Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
            return text[:300000] if len(text) > 300000 else text
        except Exception as e:
            st.error(f"Error reading DOCX {uploaded_file.name}: {e}")
            return None

    # --- TXT branch ---
    elif file_name.endswith(".txt") or "text" in content_type:
        try:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            return text[:300000] if len(text) > 300000 else text
        except Exception as e:
            st.error(f"Error reading TXT {uploaded_file.name}: {e}")
            return None

    # --- JSON branch ---
    elif file_name.endswith(".json") or "json" in content_type:
        try:
            data = json.load(uploaded_file)
            text = json.dumps(data, indent=2, ensure_ascii=False)
            return text[:300000] if len(text) > 300000 else text
        except Exception as e:
            st.error(f"Error reading JSON {uploaded_file.name}: {e}")
            return None

    else:
        st.warning(f"Unsupported file type: {uploaded_file.name}")
        return None

# ---------------------------
# Field classification (light heuristic)
# ---------------------------
JUDGMENT_PATTERNS = re.compile(
    r"\?|scale|skál|1-10|1–10|1-100|1–100|rate|how\s|igen/?nem|yes/?no",
    flags=re.IGNORECASE
)

def classify_field(field_label: str) -> str:
    """Return 'judgment' if the field looks evaluative, else 'factual'."""
    if not field_label:
        return "factual"
    return "judgment" if JUDGMENT_PATTERNS.search(field_label) else "factual"

# ---------------------------
# Build the fixed, explicit extraction prompt
# ---------------------------
def build_extraction_prompt(fields):
    """
    Deterministic, pre-written prompt:
    - Multiple entities per document
    - Semicolon-separated
    - No headers/explanations
    - Factual: leave blank if missing
    - Judgment: always provide best estimate; if impossible, 'N/A'
    - Respect specific formats requested in field text (e.g., 'Igen/Nem', '1–10')
    - Normalize numbers/dates where reasonable
    - Interpret imperfect field names; fix grammar internally (output still rows only)
    """
    cleaned = [f.strip() for f in fields if f and f.strip()]
    if not cleaned:
        return ""

    # Annotate each field with factual/judgment tag to guide the model
    annotated_lines = []
    for f in cleaned:
        tag = classify_field(f)
        if tag == "judgment":
            rule = "(judgment: always answer; if truly impossible, output 'N/A')"
        else:
            rule = "(factual: leave blank if not present)"
        annotated_lines.append(f"- {f} {rule}")

    rules = f"""
You are given the OCR/extracted text of one document. The document may contain MULTIPLE entities of the same type
(e.g., multiple invoices, CVs, etc...). Extract the requested fields for EACH entity, following these rules:

1) Output strictly one line per entity that appears in the document.
2) Separate fields with a semicolon (;). No other separators.
3) Do NOT include headers, labels, bullet points, or explanations — ONLY data rows.
4) Keep field order exactly as listed below.
5) For factual fields (numbers, dates, IDs, names, totals): if the value is not present for that entity, leave it blank.
6) For evaluative/judgment fields (e.g., ratings, predictions like Yes/No): ALWAYS provide your best estimate based on the content.
   If it is genuinely impossible to judge, output "N/A" (without quotes).
7) If a field text specifies a particular answer format (e.g., "Yes/No", "1–10 scale"), use exactly that format.
8) For numeric scales (e.g., 1–10 or 1–100), output only the integer. 
9) Be concise. Do NOT hallucinate entities that are not present.
10) If user-provided field names are colloquial or ungrammatical, interpret them correctly; the output remains rows only.

Fields to extract in this exact order:
{chr(10).join(annotated_lines)}
""".strip()

    return rules

# ---------------------------
# Run extraction with OpenAI
# ---------------------------
def extract_data_with_gpt(file_name, document_text, final_prompt, num_fields):
    """
    Sends a single message (user role) with the fixed prompt + document text.
    Parses newline-separated rows, each ';'-separated, pads/truncates to num_fields.
    """
    try:
        message = final_prompt + "\n\n<<<DOCUMENT TEXT START>>>\n" + document_text + "\n<<<DOCUMENT TEXT END>>>"
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": message}],
            max_completion_tokens=3000,
            timeout=60
        )
        raw = (response.choices[0].message.content or "").strip()
        # split rows, ignore empty lines
        rows = [r for r in (raw.split("\n")) if r.strip()]

        parsed = []
        for row in rows:
            parts = [p.strip() for p in row.split(";")]
            if all(p == "" for p in parts):
                continue
            # pad or truncate to match expected number of fields
            if len(parts) < num_fields:
                parts += [""] * (num_fields - len(parts))
            elif len(parts) > num_fields:
                parts = parts[:num_fields]
            parsed.append([file_name] + parts)
        return parsed

    except Exception as e:
        st.error(f"❌ Error processing {file_name}: {e}")
        return []

# ---------------------------
# UI
# ---------------------------
st.title("📄 General Document Extractor 47™")

st.markdown("""
Hi!
Drop in lots of PDFs or images (invoices, CVs, player profiles). We’ll get the info you want.

**Ideas you can try:**
- Invoices: number, date, totals … plus “How authentic does this look (1–10)?”
- CVs: name, email, years of experience … plus “How strong for Senior Python (1–10)?”
- Player profiles: club, matches, goals … plus “Is this player clutch? Yes/No” ⚽

Pro tip: You can ask **evaluative** things too, not just facts.
""")

st.subheader("📂 Upload your files")
uploaded_files = st.file_uploader(
    "Drop your PDFs or images here",
    type=["pdf", "jpg", "jpeg", "png", "tif", "tiff", "bmp"],
    accept_multiple_files=True
)

st.subheader("📝 What information do you want to extract?")

# Dynamic input fields
for i, val in enumerate(st.session_state.custom_fields):
    st.session_state.custom_fields[i] = st.text_input(f"Field {i+1}", val)

if st.button("➕ Add another field"):
    st.session_state.custom_fields.append("")

# --- Run Extraction ---
if st.button("🚀 Start extraction"):
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one document.")
    elif not any(f.strip() for f in st.session_state.custom_fields):
        st.warning("⚠️ Please specify at least one field to extract.")
    else:
        fields = [f.strip() for f in st.session_state.custom_fields if f.strip()]

        # Build the deterministic prompt (used internally only)
        st.session_state.final_prompt = build_extraction_prompt(fields)

        if not st.session_state.final_prompt:
            st.error("Could not build the extraction prompt. Please add at least one non-empty field.")
        else:
            # (Optional) Developer-only debug toggle. Keep commented out if you never want it visible.
            # debug_show = st.checkbox("Show generated prompt (dev only)", value=False)
            # if debug_show:
            #     st.code(
            #         st.session_state.final_prompt + "\n\n<<<DOCUMENT TEXT START>>>\n[DOCUMENT TEXT HERE]\n<<<DOCUMENT TEXT END>>>",
            #         language="markdown"
            #     )

            # Extraction loop (no prompt shown to the user)
            st.session_state.extracted_data = []
            progress = st.progress(0, text="Starting extraction...")

            total_files = len(uploaded_files)
            for idx, uploaded_file in enumerate(uploaded_files, start=1):
                file_name = uploaded_file.name
                progress.progress(idx / total_files, text=f"Processing {file_name} ({idx}/{total_files})...")
                doc_text = extract_text_from_upload(uploaded_file)
                if not doc_text:
                    continue
                rows = extract_data_with_gpt(file_name, doc_text, st.session_state.final_prompt, len(fields))
                st.session_state.extracted_data.extend(rows)

            progress.empty()

            if st.session_state.extracted_data:
                columns = ["File name"] + fields
                st.session_state.df_extracted = pd.DataFrame(st.session_state.extracted_data, columns=columns)

# Results + Excel
if not st.session_state.df_extracted.empty:
    st.write("✅ **Here are your extracted results:**")
    st.dataframe(st.session_state.df_extracted)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        st.session_state.df_extracted.to_excel(writer, sheet_name="Extracted", index=False)
    buffer.seek(0)

    st.download_button(
        label="📥 Download Excel",
        data=buffer,
        file_name="extracted_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
