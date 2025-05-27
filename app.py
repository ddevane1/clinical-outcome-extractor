#!/usr/bin/env python3
# -------- app.py (hierarchical outcome extraction) --------
import os, json, re, pdfplumber, pandas as pd, tiktoken, difflib
from openai import OpenAI
import streamlit as st

# ----- CONFIG -----
MODEL = "gpt-4o"
TOKENS_FOR_RESPONSE = 2048  # Increased for more complex structure
OVERLAP_TOKENS = 100
FUZZY_THRESHOLD = 0.75
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
        "You are an expert medical reviewer extracting clinical trial outcomes. DO NOT HALLUCINATE.\n\n"
        "CRITICAL: Identify the HIERARCHICAL STRUCTURE of outcomes from the text below:\n\n"
        "1. OUTCOME DOMAINS (broad categories):\n"
        "   - Primary outcome (usually preeclampsia-related)\n"
        "   - Secondary outcomes (listed after primary)\n"
        "   - Adverse outcomes (at different gestational ages)\n"
        "   - Stillbirth or neonatal death\n"
        "   - Death and neonatal complications\n"
        "   - Neonatal therapy\n"
        "   - Poor fetal growth\n"
        "   - Other domain headers from tables\n\n"
        "2. SPECIFIC OUTCOMES within each domain:\n"
        "   - Individual outcomes listed under each domain\n"
        "   - In tables, items indented under main headings\n"
        "   - In text, outcomes in semicolon-separated lists\n\n"
        "EXTRACTION RULES:\n"
        "- When you see 'Primary outcome:', extract it as a domain\n"
        "- When you see 'Secondary outcomes:', extract EACH listed outcome\n"
        "- For tables: main heading = domain, indented items = specific outcomes\n"
        "- Extract outcomes at ALL timepoints as separate entries\n"
        "- Use exact wording from the text\n\n"
        "Return this exact JSON structure:\n"
        "{\n"
        '  "study_info": {\n'
        '    "first_author_surname": "Author surname or None",\n'
        '    "study_design": "Study design or None",\n'
        '    "study_country": "Countries or None",\n'
        '    "patient_population": "Population description or None",\n'
        '    "targeted_condition": "Condition studied or None",\n'
        '    "diagnostic_criteria": "Diagnostic criteria or None",\n'
        '    "interventions_tested": "Intervention or None",\n'
        '    "comparison_group": "Comparator or None"\n'
        '  },\n'
        '  "outcomes": [\n'
        '    {\n'
        '      "outcome_type": "domain",\n'
        '      "outcome_name": "Primary outcome",\n'
        '      "outcome_specific": "delivery with preeclampsia before 37 weeks of gestation",\n'
        '      "definition": "defined according to ISSHP",\n'
        '      "measurement_method": "clinical diagnosis",\n'
        '      "timepoint": "before 37 weeks of gestation"\n'
        '    },\n'
        '    {\n'
        '      "outcome_type": "domain",\n'
        '      "outcome_name": "Adverse outcomes of pregnancy",\n'
        '      "outcome_specific": "",\n'
        '      "definition": "None",\n'
        '      "measurement_method": "None",\n'
        '      "timepoint": "before 34 weeks of gestation"\n'
        '    },\n'
        '    {\n'
        '      "outcome_type": "domain",\n'
        '      "outcome_name": "Death or complications",\n'
        '      "outcome_specific": "",\n'
        '      "definition": "None",\n'
        '      "measurement_method": "None",\n'
        '      "timepoint": "None"\n'
        '    },\n'
        '    {\n'
        '      "outcome_type": "specific",\n'
        '      "outcome_name": "Death or complications",\n'
        '      "outcome_specific": "Miscarriage, stillbirth, or death",\n'
        '      "definition": "None",\n'
        '      "measurement_method": "clinical records",\n'
        '      "timepoint": "None"\n'
        '    },\n'
        '    {\n'
        '      "outcome_type": "specific",\n'
        '      "outcome_name": "Death or complications",\n'
        '      "outcome_specific": "Intraventricular hemorrhage of grade ≥II",\n'
        '      "definition": "None",\n'
        '      "measurement_method": "ultrasound",\n'
        '      "timepoint": "None"\n'
        '    }\n'
        '  ]\n'
        '}\n\n'
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

        # Update study info
        chunk_study_info = data.get("study_info", {})
        for field, value in chunk_study_info.items():
            if (field not in study_info or study_info[field] == "None") and value != "None":
                study_info[field] = value

        # Collect outcomes
        all_outcomes.extend(data.get("outcomes", []))
    
    # Add pdf_name
    study_info["pdf_name"] = pdf_name
    
    # Deduplicate outcomes
    seen = set()
    unique_outcomes = []
    for outcome in all_outcomes:
        key = (
            outcome.get("outcome_type", ""),
            outcome.get("outcome_name", ""),
            outcome.get("outcome_specific", ""),
            outcome.get("timepoint", "")
        )
        if key not in seen:
            seen.add(key)
            unique_outcomes.append(outcome)
    
    return study_info, unique_outcomes

# ---------- Streamlit UI ----------
st.title("Clinical Trial Hierarchical Outcome Extractor")

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
            
            if outcomes:
                all_results.append((study_info, outcomes))
                
        progress.progress((idx + 1) / len(files))

    if all_results:
        # Convert to DataFrame format
        rows = []
        for study_info, outcomes in all_results:
            for outcome in outcomes:
                row = study_info.copy()
                row.update({
                    "outcome_type": outcome.get("outcome_type", ""),
                    "outcome_domain": outcome.get("outcome_name", ""),
                    "outcome_specific": outcome.get("outcome_specific", ""),
                    "outcome_definition": outcome.get("definition", "None"),
                    "measurement_method": outcome.get("measurement_method", "None"),
                    "timepoint": outcome.get("timepoint", "None"),
                })
                rows.append(row)
        
        df = pd.DataFrame(rows)

        # Column order
        cols = [
            "pdf_name",
            "first_author_surname",
            "study_design",
            "study_country",
            "patient_population",
            "targeted_condition",
            "diagnostic_criteria",
            "interventions_tested",
            "comparison_group",
            "outcome_type",
            "outcome_domain",
            "outcome_specific",
            "outcome_definition",
            "measurement_method",
            "timepoint",
        ]
        
        # Only use existing columns
        df = df[[col for col in cols if col in df.columns]]
        
        # Display results
        st.success(f"Extracted {len(df[df['outcome_type'] == 'domain'])} outcome domains and {len(df[df['outcome_type'] == 'specific'])} specific outcomes")
        
        # Show hierarchical view
        st.subheader("Hierarchical View")
        domains = df[df['outcome_type'] == 'domain']['outcome_domain'].unique()
        
        for domain in domains:
            domain_row = df[(df['outcome_type'] == 'domain') & (df['outcome_domain'] == domain)].iloc[0]
            st.write(f"**{domain}**")
            if domain_row['outcome_specific']:
                st.write(f"  {domain_row['outcome_specific']}")
            
            # Show specific outcomes
            specific = df[(df['outcome_type'] == 'specific') & (df['outcome_domain'] == domain)]
            for _, row in specific.iterrows():
                st.write(f"  • {row['outcome_specific']}")
                if row['outcome_definition'] != 'None':
                    st.write(f"    Definition: {row['outcome_definition']}")
                if row['timepoint'] != 'None':
                    st.write(f"    Timepoint: {row['timepoint']}")
            st.write("")
        
        # Show full table
        st.subheader("Complete Data Table")
        st.dataframe(df)
        
        # Statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Outcome Domains", len(df[df['outcome_type'] == 'domain']))
        with col2:
            st.metric("Specific Outcomes", len(df[df['outcome_type'] == 'specific']))
        
        # Download
        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "hierarchical_outcomes.csv",
            mime="text/csv"
        )
    else:
        st.error("No outcomes extracted.")