#!/usr/bin/env python3
# -------- app.py (deduplication with fuzzy matching - FIXED) --------
import os, json, re, pdfplumber, pandas as pd, tiktoken, difflib
from openai import OpenAI
import streamlit as st

# ----- CONFIG -----
MODEL = "gpt-4o"
TOKENS_FOR_RESPONSE = 1024
OVERLAP_TOKENS = 100
FUZZY_THRESHOLD = 0.85  # similarity threshold for deduping outcomes
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

# Simple normalisation for outcome names
STOPWORDS = {"the", "of", "in", "to", "for", "and", "unit", "care", "confirmed"}
def canonical(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text.lower())
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)

def is_similar(a: str, b: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold

def ask_llm(chunk: str) -> str:
    prompt = (
        "You are an expert medical reviewer. CRITICAL: DO NOT HALLUCINATE OR INVENT ANY INFORMATION.\n\n"
        "STRICT RULES:\n"
        "1. Extract ONLY information explicitly stated in the text below.\n"
        "2. Use the authors' EXACT WORDING (verbatim) - do not paraphrase.\n"
        "3. If ANY information is not explicitly stated in the text, you MUST return \"None\".\n"
        "4. Do NOT infer, assume, or guess any information.\n"
        "5. Do NOT add information from your general knowledge.\n"
        "6. If unsure whether something is stated, return \"None\".\n\n"
        "Extract the following information:\n\n"
        "• first_author_surname – surname of the first author (return \"None\" if not stated).\n"
        "• study_design – verbatim description of the study design (return \"None\" if not stated).\n"
        "• study_country – country or countries where the study was conducted (return \"None\" if not stated).\n"
        "• patient_population – verbatim description of participants (return \"None\" if not stated).\n"
        "• targeted_condition – verbatim disease or condition studied (return \"None\" if not stated).\n"
        "• diagnostic_criteria – verbatim criteria the authors used to diagnose or define the targeted condition "
        "(MUST return \"None\" if not explicitly stated).\n"
        "• interventions_tested – verbatim description of the intervention group(s) (return \"None\" if not stated).\n"
        "• comparison_group – verbatim description of the control/comparator group(s) (return \"None\" if not stated).\n"
        "• outcomes – list of outcomes; for each outcome capture:\n"
        "    • outcome_measured – concise name of the outcome (return \"None\" if not stated).\n"
        "    • outcome_definition – verbatim definition/explanation (MUST return \"None\" if authors gave no definition).\n"
        "    • measurement_method – instrument, questionnaire, lab test, etc. (return \"None\" if not stated).\n"
        "    • timepoint – when measured (e.g., 'Day 28', '12 weeks') (return \"None\" if not stated).\n\n"
        "REMEMBER: If the text does not explicitly state something, you MUST return \"None\" for that field.\n\n"
        "Return exactly this JSON structure:\n"
        "{\n"
        '  \"first_author_surname\": \"Smith\" or \"None\",\n'
        '  \"study_design\": \"Randomised controlled trial\" or \"None\",\n'
        '  \"study_country\": \"Ireland\" or \"None\",\n'
        '  \"patient_population\": \"Adults with type 2 diabetes\" or \"None\",\n'
        '  \"targeted_condition\": \"Type 2 diabetes\" or \"None\",\n'
        '  \"diagnostic_criteria\": \"HbA1c ≥ 6.5%\" or \"None\",\n'
        '  \"interventions_tested\": \"Metformin 500mg twice daily\" or \"None\",\n'
        '  \"comparison_group\": \"Placebo\" or \"None\",\n'
        '  \"outcomes\": [\n'
        "    {\n"
        '      \"outcome_measured\": \"Admission to intensive care unit\" or \"None\",\n'
        '      \"outcome_definition\": \"Admission to ICU for any cause within 30 days\" or \"None\",\n'
        '      \"measurement_method\": \"hospital record review\" or \"None\",\n'
        '      \"timepoint\": \"30 days\" or \"None\"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Text to analyze:\n"
        f"{chunk}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # Set to 0 for maximum consistency and avoiding hallucination
        max_tokens=TOKENS_FOR_RESPONSE,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

def extract_outcomes(text: str, pdf_name: str):
    max_tokens = 16000 - TOKENS_FOR_RESPONSE - 1000
    records = []
    meta = {}
    
    # Collect all outcomes from all chunks first
    all_outcomes = []
    
    for chunk in split_text(text, max_tokens):
        raw = ask_llm(chunk)
        start = raw.find("{")
        if start == -1:
            continue
        try:
            data = json.loads(raw[start:])
        except json.JSONDecodeError:
            continue

        # Update metadata from first successful chunk
        if not meta:
            meta = {
                "pdf_name": pdf_name,
                "first_author_surname": data.get("first_author_surname", "None"),
                "study_design": data.get("study_design", "None"),
                "study_country": data.get("study_country", "None"),
                "patient_population": data.get("patient_population", "None"),
                "targeted_condition": data.get("targeted_condition", "None"),
                "diagnostic_criteria": data.get("diagnostic_criteria", "None"),
                "interventions_tested": data.get("interventions_tested", "None"),
                "comparison_group": data.get("comparison_group", "None"),
            }

        # Collect all outcomes
        for o in data.get("outcomes", []):
            all_outcomes.append(o)
    
    # Now deduplicate across all collected outcomes
    seen_outcomes = []  # list of canonical outcome names we've kept
    unique_outcomes = []
    
    for o in all_outcomes:
        canon = canonical(o.get("outcome_measured", ""))
        
        # Skip if empty
        if not canon:
            continue
            
        # Fuzzy deduplication
        if any(is_similar(canon, prev) for prev in seen_outcomes):
            continue  # skip near-duplicate
            
        seen_outcomes.append(canon)
        unique_outcomes.append(o)
    
    # Create records for unique outcomes only
    for o in unique_outcomes:
        record = meta.copy()
        record.update({
            "outcome_measured": o.get("outcome_measured", "None"),
            "outcome_definition": o.get("outcome_definition", "None"),
            "measurement_method": o.get("measurement_method", "None"),
            "timepoint": o.get("timepoint", "None"),
        })
        records.append(record)

    return records

# ---------- Streamlit UI ----------
st.title("Clinical Trial Extractor – Deduplicated Outcomes")

files = st.file_uploader(
    "Upload PDF trial report(s)",
    type="pdf",
    accept_multiple_files=True
)

if files:
    rows = []
    progress = st.progress(0)
    for idx, f in enumerate(files):
        with st.spinner(f"Processing {f.name}..."):
            text = pdf_to_text(f)
            rows.extend(extract_outcomes(text, f.name))
        progress.progress((idx + 1) / len(files))

    if rows:
        df = pd.DataFrame(rows)

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
            "comparison_group",
            "outcome_measured",
            "outcome_definition",
            "measurement_method",
            "timepoint",
        ]
        df = df[desired_cols]

        st.success(f"Extracted {len(df)} unique outcomes from {len(files)} PDF(s)")
        st.dataframe(df)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "outcomes.csv",
            mime="text/csv"
        )
    else:
        st.error("No information extracted.")