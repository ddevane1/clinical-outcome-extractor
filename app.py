#!/usr/bin/env python3
# -------- app.py (deduplication with fuzzy matching - FIXED) --------
import os, json, re, pdfplumber, pandas as pd, tiktoken, difflib
from openai import OpenAI
import streamlit as st

# ----- CONFIG -----
MODEL = "gpt-4o"
TOKENS_FOR_RESPONSE = 1024
OVERLAP_TOKENS = 100
FUZZY_THRESHOLD = 0.75  # similarity threshold for deduping outcomes (lowered to catch more duplicates)
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
STOPWORDS = {"the", "of", "in", "to", "for", "and", "unit", "care", "confirmed", "due", "an", "visits", "visit", "consultations", "admission", "admissions"}
def canonical(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text.lower())
    words = [w for w in text.split() if w not in STOPWORDS]
    # Sort words to help match "hospital admission" with "admission to hospital"
    return " ".join(sorted(words))

def is_similar(a: str, b: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    # Direct similarity check
    if difflib.SequenceMatcher(None, a, b).ratio() >= threshold:
        return True
    
    # Check if one is a subset of the other (for cases like "hospital admission" vs "hospitalization")
    a_words = set(a.split())
    b_words = set(b.split())
    
    # If one set of words is a subset of the other, consider them similar
    if a_words.issubset(b_words) or b_words.issubset(a_words):
        return True
    
    # Check for common medical abbreviations
    replacements = [
        ("hospitalization", "hospital"),
        ("hospitalisation", "hospital"),
        ("icu", "intensive"),
        ("picu", "intensive"),
        ("emergency", "emergency"),
        ("ed", "emergency"),
        ("primary", "primary"),
        ("gp", "primary")
    ]
    
    a_normalized = a
    b_normalized = b
    for long_form, short_form in replacements:
        a_normalized = a_normalized.replace(long_form, short_form)
        b_normalized = b_normalized.replace(long_form, short_form)
    
    if difflib.SequenceMatcher(None, a_normalized, b_normalized).ratio() >= threshold:
        return True
    
    return False

def ask_llm(chunk: str) -> str:
    prompt = (
        "You are an expert medical reviewer. CRITICAL: DO NOT HALLUCINATE OR INVENT ANY INFORMATION.\n\n"
        "IMPORTANT INSTRUCTIONS:\n"
        "1. Extract ONLY information explicitly stated in the text below.\n"
        "2. Use the authors' EXACT WORDING (verbatim) when possible.\n"
        "3. If ANY information is not found in the text, you MUST return \"None\".\n"
        "4. For measurement methods: Look CAREFULLY for HOW outcomes were measured. This includes:\n"
        "   - Data sources (hospital records, primary care records, emergency records, claims data)\n"
        "   - Assessment tools (questionnaires, scales, lab tests)\n"
        "   - Review methods (chart review, database query, patient interview)\n"
        "   - ANY description of how the outcome was ascertained or measured\n"
        "5. For timepoints: Look for WHEN outcomes were measured (e.g., baseline, day 7, week 4, monthly, at discharge, etc.)\n"
        "6. If an outcome is measured at multiple timepoints, list ALL timepoints (e.g., \"baseline, 4 weeks, 12 weeks\")\n"
        "7. The measurement method might be described in a different paragraph from the outcome - look carefully!\n\n"
        "Extract the following information:\n\n"
        "STUDY-LEVEL INFORMATION:\n"
        "• first_author_surname – surname of the first author (return \"None\" if not stated).\n"
        "• study_design – verbatim description of the study design (return \"None\" if not stated).\n"
        "• study_country – country or countries where the study was conducted (return \"None\" if not stated).\n"
        "• patient_population – verbatim description of participants (return \"None\" if not stated).\n"
        "• targeted_condition – verbatim disease or condition studied (return \"None\" if not stated).\n"
        "• diagnostic_criteria – verbatim criteria used to diagnose/define the condition (return \"None\" if not stated).\n"
        "• interventions_tested – verbatim description of the intervention group(s) (return \"None\" if not stated).\n"
        "• comparison_group – verbatim description of the control/comparator group(s) (return \"None\" if not stated).\n\n"
        "OUTCOME-LEVEL INFORMATION:\n"
        "• outcomes – list of ALL outcomes mentioned; for each outcome capture:\n"
        "    • outcome_measured – name of the outcome (e.g., mortality, hospital admission, pain score).\n"
        "    • outcome_definition – how the authors defined this outcome (return \"None\" if no definition given).\n"
        "    • measurement_method – HOW it was measured. This is CRITICAL - look for:\n"
        "        - Where data came from (hospital records, primary care records, claims database, etc.)\n"
        "        - What tool was used (specific questionnaire, lab test, clinical assessment)\n"
        "        - How it was collected (chart review, patient interview, automated query)\n"
        "        If you can't find ANY information about how it was measured, only then return \"None\"\n"
        "    • timepoint – WHEN it was measured (e.g., \"28 days\", \"baseline and 12 weeks\", \"daily for 7 days\").\n\n"
        "IMPORTANT: Measurement methods are often described separately from outcome names. Look throughout the text!\n\n"
        "Return exactly this JSON structure:\n"
        "{\n"
        '  \"first_author_surname\": \"Smith\" or \"None\",\n'
        '  \"study_design\": \"Randomised controlled trial\" or \"None\",\n'
        '  \"study_country\": \"Ireland\" or \"None\",\n'
        '  \"patient_population\": \"Adults aged 18-65 with confirmed COVID-19\" or \"None\",\n'
        '  \"targeted_condition\": \"COVID-19\" or \"None\",\n'
        '  \"diagnostic_criteria\": \"RT-PCR confirmed SARS-CoV-2\" or \"None\",\n'
        '  \"interventions_tested\": \"Remdesivir 200mg IV day 1, then 100mg daily\" or \"None\",\n'
        '  \"comparison_group\": \"Placebo\" or \"None\",\n'
        '  \"outcomes\": [\n'
        "    {\n"
        '      \"outcome_measured\": \"All-cause mortality\",\n'
        '      \"outcome_definition\": \"Death from any cause\" or \"None\",\n'
        '      \"measurement_method\": \"Medical records review\",\n'
        '      \"timepoint\": \"28 days\"\n'
        "    },\n"
        "    {\n"
        '      \"outcome_measured\": \"Oxygen saturation\",\n'
        '      \"outcome_definition\": \"None\",\n'
        '      \"measurement_method\": \"Pulse oximetry\",\n'
        '      \"timepoint\": \"Baseline, day 1, day 3, day 5, day 7\"\n'
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
    study_info = {}
    
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

        # Update study-level metadata (take first non-None value for each field)
        if not study_info or any(v == "None" for v in study_info.values()):
            for field in ["first_author_surname", "study_design", "study_country", 
                         "patient_population", "targeted_condition", "diagnostic_criteria",
                         "interventions_tested", "comparison_group"]:
                current_value = study_info.get(field, "None")
                new_value = data.get(field, "None")
                # Update if current is None or empty and new value is not None
                if (current_value == "None" or not current_value) and new_value != "None":
                    study_info[field] = new_value

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
        is_duplicate = False
        for idx, prev in enumerate(seen_outcomes):
            if is_similar(canon, prev):
                # If it's a duplicate, merge the information
                existing = unique_outcomes[idx]
                new_timepoint = o.get("timepoint", "None")
                existing_timepoint = existing.get("timepoint", "None")
                
                # Merge timepoints if different and both are not None
                if new_timepoint != "None" and existing_timepoint != "None" and new_timepoint != existing_timepoint:
                    # Combine unique timepoints
                    all_timepoints = set()
                    for tp in [existing_timepoint, new_timepoint]:
                        all_timepoints.update([t.strip() for t in tp.split(",")])
                    existing["timepoint"] = ", ".join(sorted(all_timepoints))
                elif existing_timepoint == "None" and new_timepoint != "None":
                    existing["timepoint"] = new_timepoint
                
                # Update measurement method - prefer non-None value
                new_method = o.get("measurement_method", "None")
                existing_method = existing.get("measurement_method", "None")
                if existing_method == "None" and new_method != "None":
                    existing["measurement_method"] = new_method
                elif new_method != "None" and existing_method != "None" and new_method != existing_method:
                    # If both have methods but different, keep the more detailed one
                    if len(new_method) > len(existing_method):
                        existing["measurement_method"] = new_method
                
                # Update definition - prefer non-None value
                new_def = o.get("outcome_definition", "None")
                existing_def = existing.get("outcome_definition", "None")
                if existing_def == "None" and new_def != "None":
                    existing["outcome_definition"] = new_def
                elif new_def != "None" and existing_def != "None" and new_def != existing_def:
                    # If both have definitions but different, keep the more detailed one
                    if len(new_def) > len(existing_def):
                        existing["outcome_definition"] = new_def
                
                # Update outcome name - prefer the more detailed/specific one
                new_name = o.get("outcome_measured", "")
                existing_name = existing.get("outcome_measured", "")
                if len(new_name) > len(existing_name):
                    existing["outcome_measured"] = new_name
                    
                is_duplicate = True
                break
        
        if not is_duplicate:
            seen_outcomes.append(canon)
            unique_outcomes.append(o)
    
    # Add pdf_name to study_info
    study_info["pdf_name"] = pdf_name
    
    return study_info, unique_outcomes

# ---------- Streamlit UI ----------
st.title("Clinical Trial Extractor – Deduplicated Outcomes")

files = st.file_uploader(
    "Upload PDF trial report(s)",
    type="pdf",
    accept_multiple_files=True
)

if files:
    all_results = []
    progress = st.progress(0)
    
    for idx, f in enumerate(files):
        with st.spinner(f"Processing {f.name}..."):
            text = pdf_to_text(f)
            study_info, outcomes = extract_outcomes(text, f.name)
            
            # Store results as tuple of study info and outcomes
            if outcomes:
                all_results.append((study_info, outcomes))
                
        progress.progress((idx + 1) / len(files))

    if all_results:
        # Convert to rows for DataFrame
        rows = []
        for study_info, outcomes in all_results:
            for outcome in outcomes:
                row = study_info.copy()
                row.update({
                    "outcome_measured": outcome.get("outcome_measured", "None"),
                    "outcome_definition": outcome.get("outcome_definition", "None"),
                    "measurement_method": outcome.get("measurement_method", "None"),
                    "timepoint": outcome.get("timepoint", "None"),
                })
                rows.append(row)
        
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
        
        # Only use columns that actually exist in the DataFrame
        available_cols = [col for col in desired_cols if col in df.columns]
        df = df[available_cols]

        # Display summary
        st.success(f"Extracted {len(df)} unique outcomes from {len(files)} PDF(s)")
        
        # Show the complete table
        st.subheader("Extracted Clinical Trial Data")
        st.dataframe(df)
        
        st.download_button(
            "Download Complete CSV",
            df.to_csv(index=False),
            "outcomes.csv",
            mime="text/csv"
        )
    else:
        st.error("No information extracted.")