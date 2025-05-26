#!/usr/bin/env python3
# -------- app.py (final consolidated version) --------
import os, json, pdfplumber, pandas as pd, tiktoken
from openai import OpenAI
import streamlit as st

# ----- CONFIG -----
MODEL = "gpt-4o"             # change to gpt-4o-mini or gpt-3.5-turbo if needed
TOKENS_FOR_RESPONSE = 1024
OVERLAP_TOKENS = 100
DROP_DUPLICATE_OUTCOMES = True   # set False if you want raw rows
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
        "From the text below, extract the following information. Use the authors' "
        "exact wording (verbatim) whenever possible:\n\n"
        "• first_author_surname – surname of the first author.\n"
        "• study_design – verbatim description of the study design.\n"
        "• study_country – country or countries where the study was conducted.\n"
        "• patient_population – verbatim description of participants.\n"
        "• targeted_condition – verbatim disease or condition studied.\n"
        "• diagnostic_criteria – verbatim criteria the authors used to diagnose or define the targeted condition "
        "(write \"None\" if not stated).\n"
        "• interventions_tested – verbatim description of the interventions.\n"
        "• outcomes – list; for each outcome capture:\n"
        "    • outcome_measured – concise name of the outcome.\n"
        "    • outcome_definition – verbatim definition/explanation (write \"None\" if authors gave none).\n"
        "    • measurement_method – instrument, questionnaire, lab test, etc.\n"
        "    • timepoint – when measured (e.g., 'Day 28', '12 weeks').\n\n"
        "Return exactly this JSON structure (no extra keys):\n"
        "{\n"
        '  \"first_author_surname\": \"Surname\",\n'
        '  \"study_design\": \"Randomised controlled trial\",\n'
        '  \"study_country\": \"Ireland\",\n'
        '  \"patient_population\": \"Adults with type 2 diabetes\",\n'
        '  \"targeted_condition\": \"Type 2 diabetes\",\n'
        '  \"diagnostic_criteria\": \"HbA1c ≥ 6.5%\",\n'
        '  \"interventions_tested\": \"Metformin vs placebo\",\n'
        '  \"outcomes\": [\n'
        "    {\n"
        '      \"outcome_measured\": \"Change in HbA1c\",\n'
        '      \"outcome_definition\": \"The primary outcome was the change in HbA1c from baseline to 12 weeks.\",\n'
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
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

def extract_outcomes(text: str, pdf_name: str):
    max_tokens = 16000 - TOKENS_FOR_RESPONSE - 1000
    records = []
    meta = {}  # capture meta fields once

    for chunk in split_text(text, max_tokens):
        raw = ask_llm(chunk)
        start = raw.find("{")
        if start == -1:
            continue
        try:
            data = json.loads(raw[start:])
        except json.JSONDecodeError:
            continue

        # Capture meta once
        if not meta:
            meta = {
                "pdf_name": pdf_name,
                "first_author_surname": data.get("first_author_surname", ""),
                "study_design": data.get("study_design", ""),
                "study_country": data.get("study_country", ""),
                "patient_population": data.get("patient_population", ""),
                "targeted_condition": data.get("targeted_condition", ""),
                "diagnostic_criteria": data.get("diagnostic_criteria", ""),
                "interventions_tested": data.get("interventions_tested", ""),
            }

        # Append each outcome row
        for o in data.get("outcomes", []):
            record = meta.copy()
            record.update({
                "outcome_measured": o.get("outcome_measured", ""),
                "outcome_definition": o.get("outcome_definition", ""),
                "measurement_method": o.get("measurement_method", ""),
                "timepoint": o.get("timepoint", ""),
            })
            records.append(record)

    return records

# ---------- Streamlit UI ----------
st.title("Clinical Trial Extractor – Study Details & Outcomes")

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
        rows.extend(extract_outcomes(text, f.name))
        progress.progress((idx + 1) / len(files))

    if rows:
        df = pd.DataFrame(rows)

        # Optional: drop duplicate outcome rows (same outcome + timepoint)
        if DROP_DUPLICATE_OUTCOMES:
            df = df.drop_duplicates(subset=[
                "pdf_name",
                "outcome_measured",
                "timepoint"
            ])

        # Desired column order
        desired_cols = [
            "pdf_name",
            "first_author_surname",
            "study_design",
            "study_country",
            "patient_population",
            "targeted_condition",
            "diagnostic_criteria",
            "interventions_tested",
            "outcome_measured",
            "outcome_definition",
            "measurement_method",
            "timepoint",
        ]
        df = df[desired_cols]

        st.dataframe(df)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "outcomes.csv",
            mime="text/csv"
        )
    else:
        st.error("No information extracted.")