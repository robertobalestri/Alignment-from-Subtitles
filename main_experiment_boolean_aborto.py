#!/usr/bin/env python3
import os
import csv
import json
import logging
import re
import string
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Standard Library Imports ---
# (Keep standard imports as needed)

# --- Project-Specific Imports ---
# Assuming your project structure allows these imports
try:
    from src.ai_interface.AiModels import LLMType, AIModelsService
    from src.subtitles_managing.SubtitlesManagingModels import DialogueLine
    # Import the specific functions needed for parsing from subtitles_pipeline
    from src.subtitles_managing.subtitles_pipeline import _parse_srt_file, _clean_srt_text
    # If the above imports don't work, adjust the paths based on your exact project structure
except ImportError as e:
    logging.error(f"Failed to import necessary modules. Check project structure and paths: {e}")
    logging.error("Please ensure AIModelsService, LLMType, DialogueLine, _parse_srt_file, _clean_srt_text are importable.")
    exit(1)
# --- End Imports ---

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(lineno)d: %(message)s')
logger = logging.getLogger(__name__) # Use specific logger
# -----------------------------

dataset = "US" #CHANGE THIS TO US or IT

# --- Configuration Variables ---
INPUT_STRUCTURED_DIR = Path(f"subtitles_dataset_{dataset}_renamed_structured") # The Abbr/S##/E## folder
OUTPUT_EXPERIMENT_DIR = Path(f"output_experiment_gpt4_1_{dataset}")
CSV_OUTPUT_FILENAME = f"gpt_4_1_abortion_classification_results_{dataset}.csv"
LLM_TYPE_TO_USE = LLMType.GPT4_1
# --- End Configuration ---

# --- Load Base Prompt from file ---
PROMPT_FILE_PATH = Path(__file__).parent / "prompts" / "abortion_experiment_prompt.txt"

try:
    with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as f:
        base_prompt = f.read()
    logger.info(f"Prompt caricato da: {PROMPT_FILE_PATH}")
except FileNotFoundError:
    logger.error(f"File del prompt non trovato: {PROMPT_FILE_PATH}")
    # Fallback o exit, a seconda della logica desiderata
    base_prompt = "FALLBACK_PROMPT_ERROR_LOADING_FILE" # o exit(1)
except Exception as e:
    logger.error(f"Errore nel caricamento del prompt da {PROMPT_FILE_PATH}: {e}")
    base_prompt = "FALLBACK_PROMPT_ERROR_LOADING_FILE"
# --- End Prompt Loading ---

# ==============================================================================
# STEP 1: SRT to JSON Conversion Function (Modified to report failed files)
# ==============================================================================
def convert_all_srts_to_json(input_dir: Path):
    """
    Walks the input directory, finds all SRT files, and converts them to JSON
    in the same directory, skipping if the JSON file already exists.
    Uses imported _parse_srt_file and DialogueLine.
    Logs and collects files that failed conversion.
    """
    logger.info(f"Scanning for SRT files in: {input_dir}")
    srt_files_found = 0
    json_files_created = 0
    json_already_exist = 0
    conversion_errors = 0
    conversion_error_files = [] # List to store paths of failed SRT files

    for srt_path in input_dir.rglob("*.srt"): # Use rglob to search recursively
        srt_files_found += 1
        json_path = srt_path.with_suffix(".json")

        if json_path.exists():
            logger.debug(f"JSON exists, skipping: {srt_path.name}")
            json_already_exist += 1
            continue

        logger.debug(f"Attempting conversion: {srt_path.name}")
        try:
            # Use imported parsing function
            dialogue_entries: List[DialogueLine] = _parse_srt_file(srt_path)
            if not dialogue_entries:
                logger.warning(f"No entries parsed from {srt_path.name}, cannot create JSON.")
                conversion_errors += 1
                if srt_path not in conversion_error_files: # Avoid duplicates
                    conversion_error_files.append(srt_path)
                continue

            json_data = [entry.to_dict() for entry in dialogue_entries]

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Created JSON: {json_path.name}")
            json_files_created += 1

        except Exception as e:
            logger.error(f"Failed converting {srt_path.name}: {e}", exc_info=False)
            conversion_errors += 1
            if srt_path not in conversion_error_files: # Avoid duplicates
                conversion_error_files.append(srt_path)
            # Clean up partial JSON if error during write
            if json_path.exists():
                try: json_path.unlink()
                except OSError: pass

    # --- Print Summary and Error List ---
    logger.info(f"SRT to JSON Conversion Summary:")
    logger.info(f"  SRT Files Found: {srt_files_found}")
    logger.info(f"  JSON Files Created: {json_files_created}")
    logger.info(f"  JSON Already Existed: {json_already_exist}")
    logger.info(f"  Conversion Errors: {conversion_errors}")

    # Print the list of failed files if any
    if conversion_error_files:
        logger.warning("--- SRT Files That Failed Conversion ---")
        # Sort for consistency before printing
        conversion_error_files.sort()
        for failed_path in conversion_error_files:
            # Use normpath for consistent path separators in output
            logger.warning(f"  - {os.path.normpath(failed_path)}")
    elif conversion_errors > 0:
        logger.warning("Conversion errors were logged, but the list of failed files is unexpectedly empty.")
    else:
        logger.info("No errors encountered during SRT to JSON conversion.")
    # --- End Summary ---


# ==============================================================================
# STEP 2: Classification Functions (CSV Handling + LLM Call) - Unchanged
# ==============================================================================

def load_existing_results_from_csv(csv_path: Path) -> tuple[list, set]:
    """Loads existing results, returns list of rows and set of processed episode_codes."""
    existing_results = []
    processed_episode_codes = set()
    fieldnames = ['series', 'season', 'episode', 'episode_code', 'bool_response', 'explanation']

    if not csv_path.is_file():
        logger.info(f"CSV file not found at {csv_path}. Starting fresh.")
        return existing_results, processed_episode_codes

    logger.info(f"Loading existing results from: {csv_path}")
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames or any(f not in reader.fieldnames for f in fieldnames):
                logger.error(f"CSV file {csv_path} has incorrect headers. Expected: {fieldnames}. Found: {reader.fieldnames}. Starting fresh.")
                return [], set()

            for row in reader:
                episode_code = row.get('episode_code')
                if episode_code:
                    existing_results.append(row)
                    processed_episode_codes.add(episode_code)
                else:
                    logger.warning(f"Skipping row with missing 'episode_code' in {csv_path}: {row}")
        logger.info(f"Loaded {len(existing_results)} existing results. Found {len(processed_episode_codes)} processed episode codes.")
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}. Starting fresh.")
        return [], set()
    return existing_results, processed_episode_codes

def save_results_to_csv(results_data: list, output_csv_path: Path):
    """Saves classification results, overwriting the file."""
    if not results_data:
        logger.debug("No results data provided to save.")
        return

    fieldnames = ['series', 'season', 'episode', 'episode_code', 'bool_response', 'explanation']
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.debug(f"Saving {len(results_data)} total results to CSV: {output_csv_path}")
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            # Filter out potential non-dict entries just in case
            valid_results = [r for r in results_data if isinstance(r, dict)]
            if len(valid_results) < len(results_data):
                 logger.warning(f"Attempted to save invalid entries. Only saving {len(valid_results)} valid rows.")
            writer.writerows(valid_results)
        logger.info(f"Results saved to {output_csv_path}") # Log successful save
    except Exception as e:
        logger.error(f"Error writing CSV file {output_csv_path}: {e}")

def classify_episode_json(
    json_path: Path,
    series: str,
    season: int,
    episode: int,
    episode_code: str,
    service: AIModelsService,
    llm_type: LLMType
) -> Optional[Dict[str, Any]]:
    """Loads JSON, calls LLM for Yes/No classification with explanation, returns result dict or None."""
    logger.debug(f"Classifying episode: {episode_code} from {json_path.name}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            subtitles_data = json.load(f)
        if not subtitles_data:
            logger.warning(f"JSON file {json_path.name} is empty. Skipping.")
            return None

        full_text = "\n".join([entry.get("text", "") for entry in subtitles_data if isinstance(entry, dict) and entry.get("text")])
        if not full_text.strip():
            logger.warning(f"No text content in {json_path.name} after concatenation. Skipping.")
            return None

        prompt = base_prompt + full_text

        max_retries = 3
        retry_delay_seconds = 10
        llm_answer = "Error"
        llm_explanation = "Failed to get response from LLM."

        for attempt in range(max_retries):
            try:
                raw_response_str = service.call_llm(prompt, llm_type, with_json_extraction=True)
                
                # Process LLM response
                try: # This is the try block that pairs with the existing `except json.JSONDecodeError`
                    if isinstance(raw_response_str, dict):
                        response_data = raw_response_str
                    elif isinstance(raw_response_str, str):
                        # Clean the string response
                        cleaned_response_str = re.sub(r"^```json\s*|\s*```$", "", raw_response_str.strip(), flags=re.MULTILINE)
                        response_data = json.loads(cleaned_response_str) # This can raise json.JSONDecodeError
                    else:
                        # This error will be caught by the outer `except Exception as e` in the loop,
                        # leading to retries or eventual failure logging for the episode.
                        raise TypeError(f"LLM response is of unexpected type: {type(raw_response_str)}. Content: {raw_response_str}")
                    answer = str(response_data.get("answer", "")).strip().lower()
                    explanation = str(response_data.get("explanation", "")).strip()

                    if answer in ["yes", "no"]:
                        llm_answer = answer
                        llm_explanation = explanation if answer == "yes" else ""
                        logger.debug(f"LLM response for {episode_code}: Answer='{llm_answer}', Explanation='{llm_explanation}'")
                        break # Successful parse and valid answer
                    else:
                        logger.warning(f"LLM returned invalid answer '{answer}' for {episode_code}. Attempt {attempt + 1}/{max_retries}. Raw: {raw_response_str}")
                        if attempt == max_retries - 1:
                            llm_explanation = f"Invalid answer '{answer}' from LLM after retries. Raw: {raw_response_str}"
                        else:
                            time.sleep(retry_delay_seconds)
                            # continue to retry is implicit
                except json.JSONDecodeError as json_e:
                    logger.warning(f"Failed to parse LLM JSON response for {episode_code}: {json_e}. Attempt {attempt + 1}/{max_retries}. Cleaned: '{cleaned_response_str}', Raw: '{raw_response_str}'")
                    if attempt == max_retries - 1:
                        llm_explanation = f"Failed to parse LLM JSON response after retries: {json_e}. Cleaned: '{cleaned_response_str}', Raw: '{raw_response_str}'"
                    else:
                        time.sleep(retry_delay_seconds)
                        # continue to retry is implicit

            except Exception as e:
                logger.error(f"Error calling LLM for {episode_code} (attempt {attempt + 1}/{max_retries}): {e}", exc_info=True)
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay_seconds} seconds...")
                    time.sleep(retry_delay_seconds)
                else:
                    logger.error(f"LLM call failed after {max_retries} retries for {episode_code}.")
                    llm_explanation = f"LLM call failed after retries: {e}"
            
            if llm_answer != "Error": # if we got a valid response
                break

        # Ensure llm_answer is capitalized for the CSV, or remains 'Error'
        final_answer_for_csv = llm_answer.capitalize() if llm_answer in ["yes", "no"] else "Error"

        if final_answer_for_csv == "Error":
             logger.error(f"Failed to get valid classification for {episode_code} after retries. Final explanation: {llm_explanation}")

        logger.info(f"Episode {episode_code} classified as: {final_answer_for_csv}, Explanation: '{llm_explanation}'")
        return {
            'series': series,
            'season': season,
            'episode': episode,
            'episode_code': episode_code,
            'bool_response': final_answer_for_csv, # Yes/No/Error
            'explanation': llm_explanation
        }

    except FileNotFoundError:
        logger.error(f"JSON file not found: {json_path}")
        return {'series': series, 'season': season, 'episode': episode, 'episode_code': episode_code, 'bool_response': 'Error', 'explanation': f'File not found: {json_path}'}
    except json.JSONDecodeError as e_json_file:
        logger.error(f"Error decoding JSON from file: {json_path}: {e_json_file}")
        return {'series': series, 'season': season, 'episode': episode, 'episode_code': episode_code, 'bool_response': 'Error', 'explanation': f'Error decoding file JSON: {e_json_file}'}
    except Exception as e_main:
        logger.error(f"Unexpected error classifying {episode_code} from {json_path}: {e_main}", exc_info=True)
        return {'series': series, 'season': season, 'episode': episode, 'episode_code': episode_code, 'bool_response': 'Error', 'explanation': f'Unexpected error: {e_main}'}


def classify_all_episodes(input_dir: Path, output_csv_path: Path, service: AIModelsService, llm_type: LLMType):
    """
    Walks the input directory for JSON files, classifies them if not already done,
    and saves results incrementally to CSV.
    """
    logger.info("Loading existing classification results...")
    all_results, processed_episode_codes = load_existing_results_from_csv(output_csv_path)
    new_results_this_run = []
    files_processed_in_run = 0
    files_skipped_already_processed = 0
    files_classification_error = 0
    json_files_found = 0

    logger.info(f"Scanning for JSON files to classify in: {input_dir}")
    # Use rglob to find JSON files directly, assuming structure Abbr/S##/E##/*.json
    for json_path in input_dir.rglob("*.json"):
        json_files_found += 1
        try:
            # Parse metadata from path structure
            episode_code = json_path.stem
            parts = json_path.parts
            # Assumes structure like .../INPUT_DIR/CM/S01/E01/CMS01E01.json
            episode_folder_name = parts[-2] # E01
            season_folder_name = parts[-3] # S01
            series_abbr = parts[-4] # CM

            season_num = int(season_folder_name[1:])
            episode_num = int(episode_folder_name[1:])

        except (IndexError, ValueError) as e:
            logger.warning(f"Could not parse metadata from path: {json_path}. Skipping. Error: {e}")
            files_classification_error += 1 # Count as an error
            continue

        # Check if already processed
        if episode_code in processed_episode_codes:
            logger.debug(f"Episode {episode_code} already processed. Skipping.")
            files_skipped_already_processed += 1
            continue

        # Classify using LLM
        logger.info(f"--- Processing: {episode_code} ---")
        classification_result = classify_episode_json(
            json_path, series_abbr, season_num, episode_num, episode_code,
            service, llm_type
        )

        # Handle result and save incrementally
        if classification_result:
            all_results.append(classification_result)
            processed_episode_codes.add(episode_code)
            new_results_this_run.append(classification_result) # Track new results if needed
            files_processed_in_run += 1
            save_results_to_csv(all_results, output_csv_path) # Save combined results
        else:
            files_classification_error += 1
            logger.error(f"Classification failed for {episode_code}")


    logger.info(f"Episode Classification Summary:")
    logger.info(f"  JSON Files Found: {json_files_found}")
    logger.info(f"  Episodes Classified in this run: {files_processed_in_run}")
    logger.info(f"  Episodes Skipped (already in CSV): {files_skipped_already_processed}")
    logger.info(f"  Classification Errors: {files_classification_error}")
    logger.info(f"  Total results in CSV: {len(all_results)}")


# ==============================================================================
# Main Orchestration Function (Unchanged)
# ==============================================================================
def main():
    """Defines paths, initializes services, and runs the processing steps."""
    # --- Define Paths ---
    csv_output_path = OUTPUT_EXPERIMENT_DIR / CSV_OUTPUT_FILENAME
    # --- End Paths ---

    # --- Initialize Services ---
    try:
        # Use your actual imported service
        service = AIModelsService()
        logger.info(f"Initialized AI Service for model: {LLM_TYPE_TO_USE.name}")
    except Exception as service_e:
        logger.error(f"Failed to initialize AIModelsService: {service_e}", exc_info=True)
        return # Exit main if service fails
    # --- End Service Init ---


    # ================================================
    # === STEP 1: Convert SRT files to JSON ========
    # ================================================
    # Comment out this block to skip JSON conversion
    logger.info("--- Starting Step 1: SRT to JSON Conversion ---")
    convert_all_srts_to_json(INPUT_STRUCTURED_DIR)
    logger.info("--- Finished Step 1: SRT to JSON Conversion ---")
    # ================================================


    # ================================================
    # === STEP 2: Classify Episodes using JSON =====
    # ================================================
    # Comment out this block to skip classification
    logger.info("--- Starting Step 2: Episode Classification ---")
    classify_all_episodes(INPUT_STRUCTURED_DIR, csv_output_path, service, LLM_TYPE_TO_USE)
    logger.info("--- Finished Step 2: Episode Classification ---")
    # ================================================


# ==============================================================================
# Execution Guard (Unchanged)
# ==============================================================================
if __name__ == "__main__":
    logger.info("Script execution started.")

    # === Run Main Process ===
    main()
    # ========================

    logger.info("Script execution finished.")