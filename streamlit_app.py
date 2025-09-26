import streamlit as st
st.set_page_config(layout="centered")  # no wide layout
import pandas as pd
import PyPDF2
from pdf2image import convert_from_bytes
import gc
import pytesseract
from io import BytesIO
from openai import OpenAI
import traceback, sys

# --- Exception handler ---
def global_exception_handler(exc_type, exc_value, exc_traceback):
    st.error("Oops! Something went wrong:")
    st.code("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
sys.excepthook = global_exception_handler

# --- Session State ---
if "custom_fields" not in st.session_state:
    st.session_state.custom_fields = ["", "", ""]  # 3 inputs by default
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = []
if "df_extracted" not in st.session_state:
    st.session_state.df_extracted = pd.DataFrame()

# --- OpenAI setup ---
client = OpenAI(api_key=st.secrets["openai"]["OPENAI_API_KEY"])

# --- PDF text extraction ---
def extract_text_from_pdf(uploaded_file):
    file_name = uploaded_file.name
    pdf_content = ""

    # 1) Try direct text extraction
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            pdf_content += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error reading {file_name}: {e}")
        return None

    # 2) OCR fallback
    if len(pdf_content.strip()) < 100:
        pdf_content = ""
        try:
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
            num_pages = len(PyPDF2.PdfReader(BytesIO(file_bytes)).pages)

            progress = st.progress(0)
            for i in range(1, num_pages + 1):
                images = convert_from_bytes(file_bytes, dpi=150, first_page=i, last_page=i)
                text = pytesseract.image_to_string(images[0], lang="eng+hun")
                pdf_content += text + "\n"
                del images
                gc.collect()
                progress.progress(i / num_pages)
        except Exception as e:
            st.error(f"OCR error for {file_name}: {e}")
            return None

    # 3) Length limit
    if len(pdf_content) > 300000:
        st.warning(file_name + " is too long, only the first 300,000 characters will be processed.")
        pdf_content = pdf_content[:300000]

    return pdf_content

# --- Let GPT build the extraction prompt ---
def create_extraction_prompt(fields):
    try:
        system_msg = (
            "You are an assistant that designs extraction prompts. "
            "Given some fields the user wants, your job is to produce a clear, structured instruction "
            "to extract these fields from arbitrary documents (invoices, CVs, profiles, etc.). "
            "The prompt must:\n"
            "- Explain that documents may contain MULTIPLE entities.\n"
            "- Require output in semicolon-separated format.\n"
            "- Require one line per entity.\n"
            "- Forbid explanations or headers in the output.\n"
        )
        user_msg = "The user wants these fields:\n" + "\n".join(f"- {f}" for f in fields)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            max_completion_tokens=1000,
            timeout=30,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating prompt: {e}")
        return None

# --- Extraction run ---
def extract_data_with_gpt(file_name, text, final_prompt, num_fields):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # fixed
            messages=[{"role": "user", "content": final_prompt + "\n\nExtracted text:\n" + text}],
            max_completion_tokens=3000,
            timeout=30
        )
        raw_output = response.choices[0].message.content.strip()
        rows = raw_output.split("\n")

        parsed = []
        for row in rows:
            parts = [p.strip() for p in row.split(";")]
            # pad/truncate to match number of fields
            if len(parts) < num_fields:
                parts += [""] * (num_fields - len(parts))
            elif len(parts) > num_fields:
                parts = parts[:num_fields]
            parsed.append([file_name] + parts)
        return parsed
    except Exception as e:
        st.error(f"❌ Error processing {file_name}: {e}")
        return []

# --- UI ---
st.title("📄 General Document Extractor 47™")

st.markdown("""
This little app lets you throw in PDFs, images, invoices, CVs, or even your football player profiles.  
It will then fetch the info you ask for.  

💡 **Ideas of what to use it for:**  
- Extracting **invoice details** (numbers, dates, totals)  
- Parsing **CVs** (names, emails, skills, years of experience)  
- Scraping **player profiles** (stats, scores, how many goals someone missed 🙃)  
- Asking fun questions like *"How good would this candidate be for a Senior Python job?"*  
- Or *"How legendary is this footballer on a scale of 1-10?"* ⚽  
""")

st.subheader("📂 Upload your files")
uploaded_files = st.file_uploader(
    "Drop your PDFs or images here",
    type=["pdf", "jpg", "jpeg", "png"],
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
        # Step 1: Let GPT create the final prompt
        st.info("🤖 Asking GPT to craft the perfect extraction prompt...")
        final_prompt = create_extraction_prompt(fields)
        if final_prompt:
            st.text_area("Generated Prompt", final_prompt, height=250)

            # Step 2: Run extraction
            st.session_state.extracted_data = []
            progress = st.progress(0, text="Starting extraction...")
            for idx, uploaded_file in enumerate(uploaded_files, start=1):
                file_name = uploaded_file.name
                progress.progress(idx / len(uploaded_files), text=f"Processing {file_name}...")
                text = extract_text_from_pdf(uploaded_file)
                if not text:
                    continue
                rows = extract_data_with_gpt(file_name, text, final_prompt, len(fields))
                st.session_state.extracted_data.extend(rows)
            progress.empty()

            # Step 3: Show results
            if st.session_state.extracted_data:
                cols = ["File name"] + fields
                st.session_state.df_extracted = pd.DataFrame(st.session_state.extracted_data, columns=cols)

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
