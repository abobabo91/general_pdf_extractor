import streamlit as st
import pandas as pd
import numpy as np
import os
import PyPDF2
import pdf2image
import pytesseract
from io import BytesIO
import openai
from openai import OpenAI
import tiktoken

def count_tokens(text, model="gpt-4o"):
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    return len(tokens)



# Initialize session state variables if they don't exist
if 'extracted_text_from_invoice' not in st.session_state:
    st.session_state.extracted_text_from_invoice = []
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = []
if 'df_extracted' not in st.session_state:
    st.session_state.df_extracted = pd.DataFrame()
if 'number_of_tokens' not in st.session_state:
    st.session_state.number_of_tokens = 0

openai.organization = "org-i7aicv7Qc0PO4hkTCT4N2BqR"
openai.api_key = st.secrets['openai']["OPENAI_API_KEY"]



st.title("📄 Invoice Data Extractor")

st.write("1) Upload one or more **Documents (PDFs)** to extract relevant information.")
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

st.write("2) Enter the language(s) of the documents.")
languages = st.text_input(placeholder="Type here...")

st.write("3) Enter the informations you want to extract from the documents, separated by commas. Eg: Invoice number, Net amount, VAT")
extract_information = st.text_input(placeholder="Type here...")



#1) text extraction from pdf
if st.button("Extract PDFs"):  
    st.session_state.extracted_text_from_invoice = []      
    if uploaded_files:
        if len(uploaded_files) > 50:
            st.write("Parsing the first 50 files.")
        for uploaded_file in uploaded_files[:50]:
            file_name = uploaded_file.name
            pdf_content = ""

            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    pdf_content += page.extract_text() or ""
            except Exception as e:
                st.error(f"Error reading {file_name}: {e}")
                continue

            if len(pdf_content.strip()) < 100:
                pdf_content = ""
                try:
                    uploaded_file.seek(0)
                    images = pdf2image.convert_from_bytes(uploaded_file.read())
                    for img in images:
                        pdf_content += pytesseract.image_to_string(img, lang="hun")
                except Exception as e:
                    st.error(f"OCR failed for {file_name}: {e}")
                    continue
                
            if len(pdf_content) > 5000:
                st.write(file_name + " file is too big. Only parsing the first 5000 characters.")
                pdf_content = pdf_content[:5000]
            
            st.session_state.extracted_text_from_invoice.append([file_name, pdf_content])

    else:
        st.warning("⚠️ Please upload at least one PDF file.")


    #2) data extraction from text
    st.session_state.extracted_data = []
    if st.session_state.extracted_text_from_invoice:    
        for i in range(len(st.session_state.extracted_text_from_invoice)):
            file_name = st.session_state.extracted_text_from_invoice[i][0]
            pdf_content = st.session_state.extracted_text_from_invoice[i][1]
            
            gpt_prompt = ("""I send you an extract of a pdf bill invoice in hungarian. Your job is to find the final several data from the invoice: """ 
                        + pdf_content +  """. Output the following in order: 
                        1) the name of the partner, 
                        2) the invoice number, 
                        3) the date of the invoice,
                        4) the total gross amount of the full invoice, 
                        5) the total net amount of the invoice, 
                        6) the total VAT (ÁFA in hungarian) of the invoice. 
                        Output these values (1, 2 and 3 as strings, 4, 5 and 6 as integers) separated by ; and nothing else!""")
            
            try:    
                client = OpenAI(api_key=openai.api_key)
            
                response = client.chat.completions.create(
                    model='gpt-4o', 
                    messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": gpt_prompt}],
                    max_tokens = 50,
                    temperature=0,
                    timeout=30)
                
                extracted_text = response.choices[0].message.content.strip()
                
                if len(extracted_text.split(";")) != 6:
                    st.error(f"GPT-4 extraction failed for {file_name}: {e}")
                    continue        
                st.session_state.extracted_data.append([file_name] + extracted_text.split(";"))
                st.session_state.number_of_tokens += count_tokens(gpt_prompt)
            
            except Exception as e:
                st.error(f"GPT-4 extraction failed for {file_name}: {e}")
                continue
            
            st.write(file_name + " is being extracted.")

    if len(st.session_state.extracted_data) != 0:
        st.session_state.df_extracted = pd.DataFrame(st.session_state.extracted_data, columns=["Fájlnév", "Partner Név", "Számlaszám", "Számla Kelte", "Bruttó ár", "Nettó ár", "ÁFA"])
        st.session_state.df_extracted["Számlaszám"] = st.session_state.df_extracted["Számlaszám"].astype(str)
        st.session_state.df_extracted["1"] = np.nan

if len(st.session_state.df_extracted) > 0:        
    st.write("✅ **Extraction complete!** Here are the results:")
    st.dataframe(st.session_state.df_extracted)

    extract_csv = st.session_state.df_extracted.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Extract CSV", extract_csv, "extract_data.csv", "text/csv", key="download-excel-csv")
    
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        st.session_state.df_extracted.to_excel(writer, sheet_name='Sheet1', index=False)
        writer.close()
        download2 = st.download_button(
            label="📥 Download Extract Excel",
            data=buffer,
            file_name='extract_data.xlsx',
            mime='application/vnd.ms-excel'
        )


price = st.session_state.number_of_tokens * 2.5 / 1000000
st.write("The total cost of this process so far: $" + str(price))
        
def count_tokens(text, model="gpt-4o"):
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    return len(tokens)
