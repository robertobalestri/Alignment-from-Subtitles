#!/usr/bin/env python3
import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import re
import string

import pandas as pd
# Evaluation metrics from scikit-learn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(lineno)d: %(message)s')
logger = logging.getLogger(__name__)

# --- Project-Specific Imports ---
# Attempt to import custom cleaner, fallback to basic if not found
try:
    from src.subtitles_managing.subtitles_pipeline import _clean_srt_text
    USE_SRT_CLEANER = True
    logger.info("Using _clean_srt_text from project.")
except ImportError:
    logging.warning("Could not import _clean_srt_text. Using basic cleaning.")
    USE_SRT_CLEANER = False
    def _clean_srt_text(text: str) -> str:
        """Basic text cleaning function."""
        text = re.sub('<[^<]+?>', ' ', text)
        text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', text)
        text = re.sub(r'[\(\[].*?[\)\]]', ' ', text)
        # Keep basic punctuation that might separate words, remove others
        # Allow apostrophes within words
        text = re.sub(r'[^\w\s\']', ' ', text) # Keep words, whitespace, apostrophes
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text) # Consolidate whitespace
        return text

# --- Configuration Variables ---
JSON_ABORTO_DIR = Path("subtitles_aborto/json_output")
JSON_NOT_ABORTO_DIR = Path("subtitles_not_aborto/json_output")
OUTPUT_KEYWORD_DIR = Path("output_keyword_baseline")
RESULTS_CSV_FILENAME = "keyword_baseline_results.csv"
REPORT_FILENAME = "keyword_baseline_report.txt"

# !!! --- DEFINE KEYWORD LIST --- !!!
# Case will be ignored during matching. Uses word boundaries (\b).
# Add/remove words based on your domain knowledge and expected terms.
# Be cautious with ambiguous words (like 'choice', 'options', 'clinic', 'procedure' unless very confident).
ABORTION_KEYWORDS = [
    # Core Terms (Stems: abort, terminat*)
    "abortion", "abort", "aborted", "aborting", "abortion-on-demand", "methotrexate",     # Essential
    "termination", "terminate", "terminated", "terminating", # Essential (in pregnancy context)

    # Ideological/Political Terms
    "pro-choice", #"pro choice",  # Covered by pro-choice with optional hyphen below
    "prochoice",
    "pro choice",
    "anti-choice",
    "antichoice",
    "anti choice",
    "pro-life",   #"pro life",    # Covered by pro-life with optional hyphen below
    "prolife",
    "pro life",
    # Medical Procedures & Drugs - Highly Specific
    "d&c",                      # Keep, but note potential ambiguity with miscarriage management
    "dilation and curettage", # Specific full term
    "curette",
    "mifeprex",
    "vacuum aspiration",
    "d&e",
    "dilation and evacuation",
    "mifepristone",             # Specific drug
    "misoprostol",              # Specific drug (also used for other things, but highly correlated in context)
    "progesterone",
    "ru486", "ru-486",          # Older name for mifepristone
    "ru 486",
    "abortifacient",
    "morning-after pill", "morning after pill", "plan b", "plan-b" , "levonorgestrel", 
    "emergency contraception",
    "conscientious objection",
    # Organizations - Highly Specific
    "planned parenthood",

    # Circumstances / Context Phrases
    "unwanted pregnancy",       # Very strong indicator of context
]
# Compile regex patterns for efficiency and word boundary matching
# Add phrase patterns separately if needed
KEYWORD_PATTERNS = []
PHRASE_KEYWORDS = []
for kw in ABORTION_KEYWORDS:
    if " " in kw: # Handle phrases
        # Simple phrase matching (case-insensitive)
         PHRASE_KEYWORDS.append(kw.lower())
         logger.debug(f"Adding phrase keyword: {kw.lower()}")
    else: # Handle single words with word boundaries
        pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
        KEYWORD_PATTERNS.append(pattern)
        logger.debug(f"Adding word pattern: {pattern.pattern}")

# --- Label Mapping ---
LABEL_TO_INT = {"No": 0, "Yes": 1}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}
REPORT_LABELS_ORDER = ["No", "Yes"] # For consistent report order

# --- End Configuration ---


def load_data_for_keyword_matching(abortion_json_dir: Path, no_abortion_json_dir: Path) -> Tuple[List[str], List[int], List[str]]:
    """Loads full dialogue text, assigns INT labels (0/1), cleans text."""
    all_texts = []
    all_labels_int = []
    all_filenames = []
    processed_stems = set()
    dirs_to_process = { "No": no_abortion_json_dir, "Yes": abortion_json_dir }

    logger.info("Loading and preparing data for keyword matching...")
    for label_str, json_dir in dirs_to_process.items():
        label_int = LABEL_TO_INT[label_str]
        if not json_dir.is_dir(): continue
        logger.info(f"Processing directory for label '{label_str}' (ID {label_int}): {json_dir}")
        for json_path in json_dir.glob("*.json"):
            file_stem = json_path.stem
            if file_stem in processed_stems: continue
            processed_stems.add(file_stem)
            try:
                with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
                dialogue_texts = [entry.get("text", "") for entry in data if isinstance(entry, dict)]
                full_dialogue = "\n".join(filter(None, dialogue_texts))
                if not full_dialogue.strip(): continue
                # Use basic cleaning suitable for keyword search
                cleaned_text = _clean_srt_text(full_dialogue)
                if not cleaned_text: continue

                all_texts.append(cleaned_text)
                all_labels_int.append(label_int)
                all_filenames.append(file_stem)
            except Exception as e: logger.error(f"Error processing file {json_path.name}: {e}", exc_info=False)

    if not all_texts:
        logger.error("No valid text data loaded.")
        return [], [], []
    logger.info(f"Total documents loaded: {len(all_texts)}")
    return all_texts, all_labels_int, all_filenames


def classify_episodes_by_keyword(
    texts: List[str],
    keyword_patterns: List[re.Pattern],
    phrase_keywords: List[str]
    ) -> List[int]:
    """Classifies texts based on keyword presence. Returns list of predicted labels (0 or 1)."""
    predictions_int = []
    logger.info(f"Classifying {len(texts)} documents based on keywords...")
    for i, text in enumerate(texts):
        predicted_label = 0 # Default to "No"
        # Text should already be lowercase from cleaning, but ensure it
        text_lower = text.lower()

        # Check single-word patterns first
        for pattern in keyword_patterns:
            if pattern.search(text_lower): # Use search for efficiency
                predicted_label = 1 # Found keyword, predict "Yes"
                logger.debug(f"Keyword pattern '{pattern.pattern}' found in document {i}. Predicting Yes.")
                break # No need to check other keywords for this document

        # If not found yet, check phrase keywords
        if predicted_label == 0:
            for phrase in phrase_keywords:
                # Simple substring check for phrases (already lowercase)
                if phrase in text_lower:
                    predicted_label = 1
                    logger.debug(f"Keyword phrase '{phrase}' found in document {i}. Predicting Yes.")
                    break

        predictions_int.append(predicted_label)
        if (i + 1) % 10 == 0: # Log progress periodically
             logger.info(f"Processed {i+1}/{len(texts)} documents...")

    logger.info("Keyword classification complete.")
    return predictions_int


def generate_report(actual_labels_int: List[int], predicted_labels_int: List[int], output_dir: Path):
    """Calculates metrics and writes report."""
    if not actual_labels_int or not predicted_labels_int or len(actual_labels_int) != len(predicted_labels_int):
        logger.error("Cannot generate report due to invalid input labels.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / REPORT_FILENAME

    # Map back to string labels for reporting
    actual_labels_str = [INT_TO_LABEL.get(i, "Unknown") for i in actual_labels_int]
    predicted_labels_str = [INT_TO_LABEL.get(i, "Unknown") for i in predicted_labels_int]

    logger.info("Generating classification report...")
    try:
        report = classification_report(actual_labels_str, predicted_labels_str, labels=REPORT_LABELS_ORDER, zero_division=0)
        cm = confusion_matrix(actual_labels_str, predicted_labels_str, labels=REPORT_LABELS_ORDER)
        accuracy = accuracy_score(actual_labels_str, predicted_labels_str)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("--- Keyword Baseline Classification Report ---\n\n")
            f.write(f"Keywords used: {', '.join(ABORTION_KEYWORDS)}\n\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(f"Labels Order: {REPORT_LABELS_ORDER}\n")
            f.write(str(cm))
            f.write("\n\n")
            f.write("Classification Report:\n")
            f.write(report)

        logger.info(f"Report saved to {report_path}")
        print("\n--- Keyword Baseline Evaluation Results ---")
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(f"Labels Order: {REPORT_LABELS_ORDER}")
        print(cm)
        print("\nClassification Report:")
        print(report)
        print("-------------------------------------------")
        print(f"Detailed report saved to {report_path}")

    except Exception as e:
         logger.error(f"Failed to generate or write report: {e}", exc_info=True)


# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    logger.info("Starting Keyword Baseline classification script...")

    # Step 1: Load and Prepare Data
    texts, actual_labels_int, filenames = load_data_for_keyword_matching(JSON_ABORTO_DIR, JSON_NOT_ABORTO_DIR)

    # Step 2: Classify based on Keywords
    if texts:
        predicted_labels_int = classify_episodes_by_keyword(texts, KEYWORD_PATTERNS, PHRASE_KEYWORDS)

        # Step 3: Generate Report
        generate_report(actual_labels_int, predicted_labels_int, OUTPUT_KEYWORD_DIR)

        # Step 4: Save detailed results (Optional)
        if filenames and actual_labels_int and predicted_labels_int:
            actual_labels_str = [INT_TO_LABEL.get(i, "Unknown") for i in actual_labels_int]
            predicted_labels_str = [INT_TO_LABEL.get(i, "Unknown") for i in predicted_labels_int]
            results_df = pd.DataFrame({
                'filename': filenames,
                'human_label': actual_labels_str,
                'predicted_label': predicted_labels_str
            })
            csv_path = OUTPUT_KEYWORD_DIR / RESULTS_CSV_FILENAME
            try:
                results_df.to_csv(csv_path, index=False, encoding='utf-8')
                logger.info(f"Detailed prediction results saved to {csv_path}")
            except Exception as e:
                logger.error(f"Failed to save results CSV: {e}", exc_info=True)
        else:
            logger.warning("Skipping saving detailed CSV results due to missing data.")

    else:
        logger.error("Failed to load valid data. Aborting.")

    logger.info("Script finished.")