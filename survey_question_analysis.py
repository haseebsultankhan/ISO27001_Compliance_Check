import pandas as pd
from glob import glob
import os

# --- Configuration ---
ORIGINAL_DATA_PATH = "data/Original_Companies/*.csv"
SYNTHETIC_DATA_PATH = "data/Synethic_Companies/*.csv"
OUTPUT_FILENAME = "survey_question_analysis.md" # We'll create a Markdown file

# These columns are for metadata and not actual survey questions
EXCLUDED_COLUMNS = ['Company_ID', 'What industry does your business operate in?']


# --- Helper functions borrowed from your app2.py ---

def parse_multi_select_options(all_values):
    """Parse semicolon and comma-separated values and return unique options"""
    unique_options = set()
    for value in all_values:
        if pd.notna(value) and str(value).strip():
            value_str = str(value).strip()
            semicolon_parts = [part.strip() for part in value_str.split(';') if part.strip()]
            for part in semicolon_parts:
                comma_parts = [opt.strip() for opt in part.split(',') if opt.strip()]
                for opt in comma_parts:
                    clean_opt = opt.strip()
                    if clean_opt and clean_opt not in ['', 'nan', 'None']:
                        unique_options.add(clean_opt)
    return sorted(list(unique_options))

def smart_sort_options(options):
    """Smart sorting: positive first, negative middle, uncertain last"""
    positive, negative, uncertain = [], [], []
    for opt in options:
        opt_lower = opt.lower()
        if any(word in opt_lower for word in ['not sure', 'maybe', 'unknown', 'uncertain']):
            uncertain.append(opt)
        elif any(word in opt_lower for word in ['no', 'not applicable', 'never', 'none']):
            negative.append(opt)
        else:
            positive.append(opt)
    return sorted(positive) + sorted(negative) + sorted(uncertain)

def is_multi_select_question(all_answers):
    """Determine if a question allows multiple selections based on data content"""
    for answer in all_answers:
        if pd.notna(answer) and ';' in str(answer):
            return True
    return False

# --- Main Analysis Logic ---

print("Starting survey question and answer analysis...")

# 1. Load all company data into one DataFrame
try:
    print(f"Loading original companies from: {ORIGINAL_DATA_PATH}")
    original_files = glob(ORIGINAL_DATA_PATH)
    if not original_files: raise FileNotFoundError("No files found for original companies.")
    
    print(f"Loading synthetic companies from: {SYNTHETIC_DATA_PATH}")
    synthetic_files = glob(SYNTHETIC_DATA_PATH)
    if not synthetic_files: raise FileNotFoundError("No files found for synthetic companies.")

    all_companies_df = pd.concat(
        [pd.read_csv(f) for f in original_files + synthetic_files],
        ignore_index=True
    )
    print("All company data loaded successfully.")
except FileNotFoundError as e:
    print(f"\nFATAL ERROR: Could not find data files. {e}")
    print("Please make sure your 'data/Original_Companies' and 'data/Synethic_Companies' folders exist and contain CSV files.")
    exit()
except Exception as e:
    print(f"\nAn error occurred during data loading: {e}")
    exit()


# 2. Get a unique, sorted list of all questions (columns)
all_questions = sorted([
    col for col in all_companies_df.columns if col not in EXCLUDED_COLUMNS
])

print(f"Found {len(all_questions)} unique questions to analyze.")

# 3. Open the output file and write the analysis
with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
    f.write("# Survey Question and Answer Analysis\n\n")
    f.write("This document lists every survey question and all its unique answers found across all company datasets.\n\n")
    f.write("**Your Task:** For each question, review the unique answers and decide which ones should be considered 'Negative' (meaning the control is NOT met). All other answers will be considered 'Positive'.\n\n")
    f.write("---\n\n")

    # 4. Loop through each question and write its details to the file
    for i, question in enumerate(all_questions):
        f.write(f"## {i+1}. {question}\n\n")
        
        all_answers_for_question = all_companies_df[question].dropna().tolist()
        
        if not all_answers_for_question:
            f.write("*No answers found for this question.*\n\n")
            f.write("---\n\n")
            continue

        # Check if this is a multi-select question and parse accordingly
        if is_multi_select_question(all_answers_for_question) or "(select all that apply)" in question.lower():
            unique_options = parse_multi_select_options(all_answers_for_question)
            f.write("**Type:** Multi-Select (parsed from responses)\n\n")
        else:
            # For single-select, just get unique values
            unique_options = [str(opt).strip() for opt in all_companies_df[question].dropna().unique()]
            f.write("**Type:** Single-Select\n\n")

        # Smart sort the options for better readability
        sorted_unique_options = smart_sort_options(unique_options)
        
        f.write("**Unique Answers Found:**\n")
        f.write("```\n")
        for option in sorted_unique_options:
            f.write(f"- {option}\n")
        f.write("```\n\n")
        
        # This is the part for you to fill out
        f.write("**Your Analysis (Please Complete):**\n\n")
        f.write("*   **Negative Answers (control NOT met):** [List the answers from above that mean 'No' or failure to comply]\n")
        f.write("*   **Positive Answers (control IS met):** [All other answers]\n")
        f.write("*   **Notes:** [Any special considerations for this question?]\n\n")
        f.write("---\n\n")

print(f"\nAnalysis complete! Please review the generated file: '{OUTPUT_FILENAME}'")