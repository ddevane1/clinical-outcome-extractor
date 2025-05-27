#!/usr/bin/env python3
# -------- app.py (hierarchical outcome extraction) --------
import os, json, re, pdfplumber, pandas as pd, tiktoken, difflib
from openai import OpenAI
import streamlit as st

# ----- CONFIG -----
MODEL = "gpt-4o"
TOKENS_FOR_RESPONSE = 1024
OVERLAP_TOKENS = 100
FUZZY_THRESHOLD = 0.75  # similarity threshold for deduping outcomes
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

# Enhanced normalisation for outcome names
STOPWORDS = {"the", "of", "in", "to", "for", "and", "unit", "care", "confirmed", "due", "an", "visits", "visit", "consultations", "admission", "admissions", "any", "all"}
def canonical(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text.lower())
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(sorted(words))

def is_similar(a: str, b: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    # Direct similarity check
    if difflib.SequenceMatcher(None, a, b).ratio() >= threshold:
        return True
    
    # Check if one is a subset of the other
    a_words = set(a.split())
    b_words = set(b.split())
    
    if a_words.issubset(b_words) or b_words.issubset(a_words):
        return True
    
    # Check for common medical abbreviations
    replacements = [
        ("hospitalization", "hospital"),
        ("hospitalisation", "hospital"),
        ("icu", "intensive"),
        ("picu", "intensive"),
        ("nicu", "intensive"),
        ("therapy", "therapy"),
        ("therapies", "therapy"),
        ("therapeutic", "therapy"),
        ("neonatal", "neonatal"),
        ("fetal", "fetal"),
        ("foetal", "fetal"),
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
        "You are an expert medical reviewer extracting outcomes from clinical trials. CRITICAL: DO NOT HALLUCINATE OR INVENT ANY INFORMATION.\n\n"
        "IMPORTANT INSTRUCTIONS:\n"
        "1. Extract ONLY information explicitly stated in the text below.\n"
        "2. Use the authors' EXACT WORDING (verbatim) when possible.\n"
        "3. Identify the HIERARCHICAL STRUCTURE of outcomes:\n"
        "   - OUTCOME DOMAINS: Broad categories like 'Death or complications', 'Therapy', 'Poor fetal growth', 'Adverse outcomes'\n"
        "   - SPECIFIC OUTCOMES: Individual outcomes within each domain\n"
        "4. For tables with outcome categories:\n"
        "   - The main heading (e.g., 'Death or complications') is an OUTCOME DOMAIN\n"
        "   - Each item listed under it is a SPECIFIC OUTCOME within that domain\n"
        "5. Common outcome domains include:\n"
        "   - Primary outcome (usually preeclampsia-related)\n"
        "   - Adverse outcomes (at different gestational ages)\n"
        "   - Stillbirth or death\n"
        "   - Death or complications\n"
        "   - Therapy (neonatal therapy)\n"
        "   - Poor fetal growth\n"
        "6. Extract ALL outcomes mentioned, whether in text, lists, or tables\n"
        "7. For timepoints: Include specific gestational ages or time periods\n"
        "8. For measurement methods: Include how outcomes were measured/defined\n\n"
        "Extract the following information:\n\n"
        "STUDY-LEVEL INFORMATION:\n"
        "• first_author_surname – surname of the first author\n"
        "• study_design – verbatim description of the study design\n"
        "• study_country – country or countries where the study was conducted\n"
        "• patient_population – verbatim description of participants\n"
        "• targeted_condition – verbatim disease or condition studied\n"
        "• diagnostic_criteria – verbatim criteria used to diagnose/define the condition\n"
        "• interventions_tested – verbatim description of the intervention group(s)\n"
        "• comparison_group – verbatim description of the control/comparator group(s)\n\n"
        "OUTCOME INFORMATION:\n"
        "• outcome_domains – list of outcome domains/categories, for each domain capture:\n"
        "    • domain_name – name of the outcome domain (e.g., 'Death or complications', 'Therapy')\n"
        "    • domain_definition – definition of the domain if provided (or 'None')\n"
        "    • timepoint – when measured (e.g., 'before 37 weeks') if applicable\n"
        "    • specific_outcomes – list of specific outcomes within this domain:\n"
        "        • outcome_name – exact name of the specific outcome\n"
        "        • outcome_definition – definition if provided\n"
        "        • measurement_method – how it was measured\n"
        "        • timepoint – specific timing if different from domain\n\n"
        "EXAMPLE: If you see a table with:\n"
        "Death or complications — no. (%)\n"
        "  Any                                32 (4.0)    48 (5.8)\n"
        "  Miscarriage, stillbirth, or death  19 (2.4)    26 (3.2)\n"
        "  Intraventricular hemorrhage        2 (0.3)     1 (0.1)\n"
        "\n"
        "Extract as:\n"
        "domain_name: 'Death or complications'\n"
        "specific_outcomes: [\n"
        "  {outcome_name: 'Any death or complications'},\n"
        "  {outcome_name: 'Miscarriage, stillbirth, or death'},\n"
        "  {outcome_name: 'Intraventricular hemorrhage'}\n"
        "]\n\n"
        "Return exactly this JSON structure:\n"
        "{\n"
        '  \"first_author_surname\": \"Surname or None\",\n'
        '  \"study_design\": \"Study design or None\",\n'
        '  \"study_country\": \"Country or None\",\n'
        '  \"patient_population\": \"Population or None\",\n'
        '  \"targeted_condition\": \"Condition or None\",\n'
        '  \"diagnostic_criteria\": \"Criteria or None\",\n'
        '  \"interventions_tested\": \"Intervention or None\",\n'
        '  \"comparison_group\": \"Comparator or None\",\n'
        '  \"outcome_domains\": [\n'
        "    {\n"
        '      \"domain_name\": \"Death or complications\",\n'
        '      \"domain_definition\": \"None\",\n'
        '      \"timepoint\": \"None\",\n'
        '      \"specific_outcomes\": [\n'
        "        {\n"
        '          \"outcome_name\": \"Miscarriage, stillbirth, or death\",\n'
        '          \"outcome_definition\": \"None\",\n'
        '          \"measurement_method\": \"Clinical records\",\n'
        '          \"timepoint\": \"None\"\n'
        "        },\n"
        "        {\n"
        '          \"outcome_name\": \"Intraventricular hemorrhage of grade ≥II\",\n'
        '          \"outcome_definition\": \"None\",\n'
        '          \"measurement_method\": \"Ultrasound\",\n'
        '          \"timepoint\": \"None\"\n'
        "        }\n"
        "      ]\n"
        "    },\n"
        "    {\n"
        '      \"domain_name\": \"Therapy\",\n'
        '      \"domain_definition\": \"Neonatal therapy requirements\",\n'
        '      \"timepoint\": \"Until discharge\",\n'
        '      \"specific_outcomes\": [\n'
        "        {\n"
        '          \"outcome_name\": \"Admission to intensive care unit\",\n'
        '          \"outcome_definition\": \"None\",\n'
        '          \"measurement_method\": \"Hospital records\",\n'
        '          \"timepoint\": \"None\"\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Text to analyze:\n"
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
    study_info = {}
    all_domains = []
    
    for chunk in split_text(text, max_tokens):
        raw = ask_llm(chunk)
        start = raw.find("{")
        if start == -1:
            continue
        try:
            data = json.loads(raw[start:])
        except json.JSONDecodeError:
            continue

        # Update study-level metadata
        if not study_info or any(v == "None" for v in study_info.values()):
            for field in ["first_author_surname", "study_design", "study_country", 
                         "patient_population", "targeted_condition", "diagnostic_criteria",
                         "interventions_tested", "comparison_group"]:
                current_value = study_info.get(field, "None")
                new_value = data.get(field, "None")
                if (current_value == "None" or not current_value) and new_value != "None":
                    study_info[field] = new_value

        # Collect all outcome domains
        for domain in data.get("outcome_domains", []):
            all_domains.append(domain)
    
    # Deduplicate domains and merge their outcomes
    unique_domains = {}
    for domain in all_domains:
        domain_name = domain.get("domain_name", "")
        canon = canonical(domain_name)
        
        if canon in unique_domains:
            # Merge specific outcomes
            existing = unique_domains[canon]
            existing_outcomes = {canonical(o.get("outcome_name", "")): o 
                               for o in existing.get("specific_outcomes", [])}
            
            for outcome in domain.get("specific_outcomes", []):
                outcome_canon = canonical(outcome.get("outcome_name", ""))
                if outcome_canon not in existing_outcomes:
                    existing["specific_outcomes"].append(outcome)
                else:
                    # Update with non-None values
                    for field in ["outcome_definition", "measurement_method", "timepoint"]:
                        if existing_outcomes[outcome_canon].get(field) == "None" and outcome.get(field) != "None":
                            existing_outcomes[outcome_canon][field] = outcome.get(field)
        else:
            unique_domains[canon] = domain
    
    # Add pdf_name to study_info
    study_info["pdf_name"] = pdf_name
    
    return study_info, list(unique_domains.values())

# ---------- Streamlit UI ----------
st.title("Clinical Trial Outcome Extractor – Hierarchical Structure")

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
            study_info, domains = extract_outcomes(text, f.name)
            
            if domains:
                all_results.append((study_info, domains))
                
        progress.progress((idx + 1) / len(files))

    if all_results:
        # Convert to rows for DataFrame
        rows = []
        for study_info, domains in all_results:
            for domain in domains:
                # Add domain-level row
                domain_row = study_info.copy()
                domain_row.update({
                    "outcome_level": "DOMAIN",
                    "outcome_domain": domain.get("domain_name", "None"),
                    "outcome_name": "",
                    "outcome_definition": domain.get("domain_definition", "None"),
                    "measurement_method": "",
                    "timepoint": domain.get("timepoint", "None"),
                })
                rows.append(domain_row)
                
                # Add specific outcome rows
                for outcome in domain.get("specific_outcomes", []):
                    outcome_row = study_info.copy()
                    outcome_row.update({
                        "outcome_level": "SPECIFIC",
                        "outcome_domain": domain.get("domain_name", "None"),
                        "outcome_name": outcome.get("outcome_name", "None"),
                        "outcome_definition": outcome.get("outcome_definition", "None"),
                        "measurement_method": outcome.get("measurement_method", "None"),
                        "timepoint": outcome.get("timepoint", "None"),
                    })
                    rows.append(outcome_row)
        
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
            "outcome_level",
            "outcome_domain",
            "outcome_name",
            "outcome_definition",
            "measurement_method",
            "timepoint",
        ]
        
        # Only use columns that actually exist
        available_cols = [col for col in desired_cols if col in df.columns]
        df = df[available_cols]

        # Display summary
        st.success(f"Extracted {len(df[df['outcome_level'] == 'DOMAIN'])} outcome domains with {len(df[df['outcome_level'] == 'SPECIFIC'])} specific outcomes from {len(files)} PDF(s)")
        
        # Show hierarchical view
        st.subheader("Hierarchical View of Outcomes")
        
        # Group by domain
        for domain_name in df[df['outcome_level'] == 'DOMAIN']['outcome_domain'].unique():
            st.write(f"**{domain_name}**")
            specific_outcomes = df[(df['outcome_level'] == 'SPECIFIC') & (df['outcome_domain'] == domain_name)]
            if not specific_outcomes.empty:
                for _, row in specific_outcomes.iterrows():
                    st.write(f"  • {row['outcome_name']}")
                    if row['outcome_definition'] != 'None':
                        st.write(f"    Definition: {row['outcome_definition']}")
                    if row['measurement_method'] != 'None':
                        st.write(f"    Method: {row['measurement_method']}")
                    if row['timepoint'] != 'None':
                        st.write(f"    Timepoint: {row['timepoint']}")
            st.write("")  # Empty line between domains
        
        # Show full table
        st.subheader("Complete Extracted Data")
        st.dataframe(df)
        
        # Download button
        st.download_button(
            "Download Complete CSV",
            df.to_csv(index=False),
            "hierarchical_outcomes.csv",
            mime="text/csv"
        )
    else:
        st.error("No information extracted.")