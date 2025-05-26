#!/usr/bin/env python3
# -------- app.py (GPT‑4o with debug prints) --------
import os, json, pdfplumber, pandas as pd, tiktoken
from openai import OpenAI
import streamlit as st

# ----- CONFIG -----
MODEL = "gpt-4o"             # switch to "gpt-4o-mini" or "gpt-3.5-turbo" if needed
TOKENS_FOR_RESPONSE = 1024
OVERLAP_TOKENS = 100
# -------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
enc = tiktoken.encoding_for_model(MODEL)

def pdf_to_text(file):
    with pdfplumber.open(file) as pdf:
        return "\n\n".join(p.extract_text() or "" for p in pdf.pages)

def num_tokens(text: str) -> int:
    return len(enc.encode(text))

def split_text(text, limit, overlap=OVERLAP_TOKENS):
    words, chunk, size = text.split(), [], 0
    for w in words:
        wlen = num_tokens(w + " ")
        if size + wlen > limit:
            yield " ".join(chunk)
            chunk = chunk[-overlap:] if overlap else []
            size = num_tokens(" ".join(chunk))
        chunk.append(w)
        size += wlen
    if chunk:
        yield " ".join(chunk)

def ask_llm(chunk: str) -> str:
    prompt = (
        "You are an expert medical reviewer.\n"
        "From the text below, extract the following information:\n"
        "1. Surname of the last author listed on the paper.\n"
        "2. Year of publication (four digits).\n"
        "3. For each clinical outcome, extract:\n"
        "   • description – concise name of the outcome.\n"
        "   • definition_verbatim – full sentence(s) from the paper that define or explain the outcome.\n"
        "   • measurement_method – instrument, questionnaire, lab test, etc.\n"
        "   • timepoint – when it was measured (e.g., 'Day 28', '12 weeks').\n\n"
        "Return exactly this JSON structure (no extra keys):\n"
        "{\n"
        '  \"last_author_surname\": \"Surname\",\n'
        '  \"publication_year\": 2024,\n'
        '  \"outcomes\": [\n'
        "    {\n"
        '      \"description\": \"Change in HbA1c\",\n'
        '      \"definition_verbatim\": \"The primary outcome was the change in HbA1c from baseline to 12 weeks.\",\n'
        '      \"measurement_method\": \"blood test\",\n'
        '      \"timepoint\": \"12 weeks\"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Text:\n"
        f"{chunk}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=TOKENS_FOR_RESPONSE,
        response_format={"type": "json_object"}  # strict JSON
    )
    return response.choices[0].message.content

def extract_outcomes(text: str):
    max_tokens = 16000 - TOKENS_FOR_RESPONSE - 1000
    records = []
    meta = {}

    for idx, chunk in enumerate(split_text(text, max_tokens)):
        raw = ask_llm(chunk)

        # ---- DEBUG PRINTS ----
        print(f"\n=== RAW RESPONSE for chunk {idx+1} (first 1000 chars) ===\n")
        print(raw[:1000])
        print("\n=== END RAW RESPONSE ===\n")
        # ----------------------

        start = raw.find("{")
        if start == -1:
            continue
        try:
            data = json.loads(raw[start:])
        except json.JSONDecodeError as e:
            print("JSONDecodeError:", e)
            continue

        if not meta and "last_author_surname" in data:
            meta["last_author_surname"] = data.get("last_author_surname", "")
            meta["publication_year"] = data.get("publication_year", "")

        for o in data.get("outcomes", []):
            records.append({
                "last_author_surname": meta.get("last_author_surname", ""),
                "publication_year": meta.get("publication_year", ""),
                "description": o.get("description", ""),
                "definition_verbatim": o.get("definition_verbatim", ""),
                "measurement_method": o.get("measurement_method", ""),
                "timepoint": o.get("timepoint", ""),
            })
    return records

# ---------- Streamlit UI ----------
st.title("LLM‑Driven Outcome Extractor (GPT‑4o, debug mode)")

files = st.file_uploader(
    "Upload PDF trial report(s)",
    type="pdf",
    accept_multiple_files=True
)

if files:
    rows = []
    progress = st.progress(0)
    for idx, f in enumerate(files):
        text = pdf_to_text(f)
        rows.extend(extract_outcomes(text))
        progress.progress((idx + 1) / len(files))

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "outcomes.csv",
            mime="text/csv"
        )
    else:
        st.error("No outcomes extracted.")