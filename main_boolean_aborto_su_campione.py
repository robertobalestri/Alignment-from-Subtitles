import json
import os
import csv
from collections import Counter
from src.subtitles_managing.subtitles_pipeline import process_subtitles_folder_to_jsons
from src.ai_interface.AiModels import LLMType, AIModelsService # Assuming these are correctly defined
from pathlib import Path
import logging
import string
import time # Import time for delays

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(filename)s:%(lineno)d: %(message)s')
# -----------------------------

# --- Configuration Variables ---
USE_BEST_OF_THREE = False # Set to True to enable best-of-three voting

# SELECT YOUR MODEL HERE:
llm_type_var = LLMType.GPT4_1 # Or LLMType.GPT4O, LLMType.GEMINI25, etc.

# Define the *name* of the CSV file globally
CSV_OUTPUT_FILENAME = "llm_classification_results_WITH_DOUBTS_GPT4_1_BESTOF1_TEMP_0_3.csv"
# -----------------------------


# --- Load Base Prompt from file ---
PROMPT_FILE_PATH = Path(__file__).parent / "prompts" / "abortion_sample_prompt.txt"

try:
    with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as f:
        base_prompt = f.read()
    logging.info(f"Prompt caricato da: {PROMPT_FILE_PATH}")
except FileNotFoundError:
    logging.error(f"File del prompt non trovato: {PROMPT_FILE_PATH}")
    base_prompt = "FALLBACK_PROMPT_ERROR_LOADING_FILE" # o exit(1)
except Exception as e:
    logging.error(f"Errore nel caricamento del prompt da {PROMPT_FILE_PATH}: {e}")
    base_prompt = "FALLBACK_PROMPT_ERROR_LOADING_FILE"
# --- End Prompt Loading ---

# ==============================================================================
# Helper Function for Best-of-Three Prediction
# ==============================================================================
def get_best_of_three_prediction(service: AIModelsService, prompt: str, llm_type: LLMType) -> str:
    """Calls the LLM three times and returns the majority vote ('Yes' or 'No')."""
    predictions = []
    valid_responses = ["yes", "no"]
    logging.debug(f"Initiating best-of-three prediction...")

    for i in range(3):
        try:
            logging.debug(f"  Best-of-three call {i+1}/3...")
            response = service.call_llm(prompt, llm_type, with_json_extraction=False)
            prediction = str(response).strip().lower()
            prediction = prediction.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation

            if prediction in valid_responses:
                predictions.append(prediction)
                logging.debug(f"    Call {i+1} prediction: {prediction}")
            else:
                logging.warning(f"    Call {i+1} returned invalid response: '{response}'. Ignoring.")
        except Exception as e:
            logging.error(f"    Error during LLM call {i+1}/3: {e}. Ignoring this call.")

    if not predictions:
        logging.error("Best-of-three: No valid predictions received.")
        return "Error" # Indicate error

    vote_counts = Counter(predictions)
    most_common = vote_counts.most_common(1)
    final_prediction = most_common[0][0]
    logging.debug(f"  Best-of-three votes: {vote_counts}. Final prediction: {final_prediction}")
    return final_prediction

# ==============================================================================
# Function to Load Existing Results from CSV
# ==============================================================================
def load_existing_results(csv_path: Path) -> tuple[list, set]:
    """
    Loads existing results from a CSV file.

    Args:
        csv_path: The Path object for the input CSV file.

    Returns:
        A tuple containing:
        - list: A list of dictionaries representing existing results.
        - set: A set of filenames (stems) that have already been processed.
    """
    existing_results = []
    processed_filenames = set()
    fieldnames = ['filename', 'actual', 'predicted'] # Expected headers

    if not csv_path.is_file():
        logging.info(f"CSV file not found at {csv_path}. Starting fresh.")
        return existing_results, processed_filenames

    logging.info(f"Loading existing results from: {csv_path}")
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # Check if headers match expected fieldnames
            if not reader.fieldnames or any(f not in reader.fieldnames for f in fieldnames):
                 logging.error(f"CSV file {csv_path} has missing or incorrect headers. Expected: {fieldnames}. Found: {reader.fieldnames}. Cannot load existing results.")
                 # Return empty, effectively starting fresh but warning the user
                 return [], set()

            for row in reader:
                # Basic validation: Ensure essential keys exist and filename is not empty
                if row.get('filename'):
                    existing_results.append(row)
                    processed_filenames.add(row['filename']) # Add filename stem to the set
                else:
                     logging.warning(f"Skipping row with missing filename in {csv_path}: {row}")

        logging.info(f"Loaded {len(existing_results)} existing results. Found {len(processed_filenames)} processed filenames.")
    except FileNotFoundError:
        logging.info(f"CSV file not found at {csv_path}. Starting fresh.") # Should be caught by is_file(), but belt-and-suspenders
    except Exception as e:
        logging.error(f"Error reading CSV file {csv_path}: {e}. Proceeding without existing results.")
        return [], set() # Return empty on error to avoid data corruption

    return existing_results, processed_filenames

# ==============================================================================
# Function to Save Results to CSV
# ==============================================================================
def save_results_to_csv(results_data: list, output_csv_path: Path):
    """
    Saves the classification results to a CSV file. Overwrites existing file.

    Args:
        results_data: A list of dictionaries, where each dict has
                      'filename', 'actual', and 'predicted' keys.
        output_csv_path: The Path object for the output CSV file.
    """
    if not results_data:
        logging.warning("No results data (new or old) provided to save to CSV.")
        return

    fieldnames = ['filename', 'actual', 'predicted']
    # Ensure parent directory exists
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logging.info(f"Saving {len(results_data)} total classification results to CSV: {output_csv_path}")
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            # Filter out any potential None entries just in case
            valid_results = [r for r in results_data if isinstance(r, dict)]
            if len(valid_results) < len(results_data):
                 logging.warning(f"Found {len(results_data) - len(valid_results)} invalid entries in results data. Saving only valid ones.")
            writer.writerows(valid_results)
        logging.info(f"Successfully saved results to {output_csv_path}")
    except IOError as e:
        logging.error(f"Error writing CSV file {output_csv_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while writing CSV: {e}")


# ==============================================================================
# Function to Calculate and Print Confusion Matrix from Collected Data
# ==============================================================================
# (This function remains the same as before, as it takes the list of results)
def calculate_and_print_confusion_matrix(results_data: list):
    """
    Calculates and prints a confusion matrix based on the collected results data.

    Args:
        results_data: A list of dictionaries, where each dict has
                      'filename', 'actual', and 'predicted' keys.
    """
    tp = 0  # True Positives: Actual=Yes, Predicted=Yes
    fn = 0  # False Negatives: Actual=Yes, Predicted=No
    tn = 0  # True Negatives: Actual=No, Predicted=No
    fp = 0  # False Positives: Actual=No, Predicted=Yes
    errors = 0 # Count predictions that were neither 'yes' nor 'no'

    print("\n" + "="*30)
    print("Analyzing Collected Results for Confusion Matrix")
    print("="*30)

    valid_results_count = 0
    if not results_data:
        logging.warning("No results data provided for confusion matrix calculation.")
        print("No results to analyze.")
        print("="*30)
        return

    for result in results_data:
         # Check if result is a dictionary and has the required keys
         if not isinstance(result, dict) or not all(k in result for k in ['filename', 'actual', 'predicted']):
              logging.warning(f"Skipping invalid result entry: {result}")
              continue

         valid_results_count += 1
         actual = str(result.get('actual', '')).lower() # Ensure string and lowercase
         predicted = str(result.get('predicted', '')).lower() # Ensure string and lowercase

         if actual == "yes":
             if predicted == "yes":
                 tp += 1
             elif predicted == "no":
                 fn += 1
             else:
                 errors += 1
                 logging.warning(f"Invalid prediction '{predicted}' for actual 'Yes' file: {result.get('filename')}")
         elif actual == "no":
             if predicted == "no":
                 tn += 1
             elif predicted == "yes":
                 fp += 1
             else:
                 errors += 1
                 logging.warning(f"Invalid prediction '{predicted}' for actual 'No' file: {result.get('filename')}")
         else:
             logging.error(f"Invalid actual value '{actual}' found in results for file: {result.get('filename')}")
             errors += 1 # Count this as an error too


    print(f"Total valid results analyzed: {valid_results_count}")
    # --- Print Results ---
    print("\n--- Raw Counts ---")
    print(f"True Positives (Actual=Yes, Predicted=Yes): {tp}")
    print(f"False Negatives (Actual=Yes, Predicted=No): {fn}")
    print(f"True Negatives (Actual=No, Predicted=No):  {tn}")
    print(f"False Positives (Actual=No, Predicted=Yes): {fp}")
    if errors > 0:
         print(f"Predictions with errors/invalid format/invalid actual: {errors}")

    print("\n--- Confusion Matrix ---")
    print("                  Predicted")
    print("                 No     Yes")
    print("Actual No    | {:^5d} | {:^5d} |".format(tn, fp))
    print("Actual Yes   | {:^5d} | {:^5d} |".format(fn, tp))
    print("="*30 + "\n")

    # --- Calculate Metrics ---
    total_valid_predictions = tp + fn + tn + fp # Exclude errors from metrics calculation base

    if total_valid_predictions > 0:
        accuracy = (tp + tn) / total_valid_predictions if total_valid_predictions > 0 else 0
        print(f"Accuracy:  {accuracy:.4f}  ({tp + tn}/{total_valid_predictions})")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision_str = f"{precision:.4f}  ({tp}/{tp + fp})" if (tp + fp) > 0 else "N/A (No positive predictions)"
        print(f"Precision: {precision_str}")

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_str = f"{recall:.4f}  ({tp}/{tp + fn})" if (tp + fn) > 0 else "N/A (No actual positives classified)"
        print(f"Recall:    {recall_str}")

        if (precision + recall) > 0:
             f1_score = 2 * (precision * recall) / (precision + recall)
             print(f"F1-Score:  {f1_score:.4f}")
        else:
             print("F1-Score:  N/A")
    else:
        print("No valid predictions found to calculate metrics.")

    print("="*30)

# ==============================================================================
# Function to process a single JSON file (refactored to return result)
# ==============================================================================
def process_json_file(json_file_path: Path, actual_category: str, service: AIModelsService, llm_type: LLMType) -> dict | None:
    """Loads JSON, calls LLM, cleans prediction, and returns the result dict or None."""
    file_stem = json_file_path.stem
    logging.debug(f"Processing (Actual {actual_category}): {json_file_path.name}")

    try:
        # Load subtitles
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract dialogue texts
        dialogue_texts = [entry["text"] for entry in data if isinstance(entry, dict) and "text" in entry]
        if not dialogue_texts:
             logging.warning(f"No dialogue text found in {json_file_path.name}. Skipping.")
             return None # Skip this file if no text

        joined_dialogue = "\n".join(dialogue_texts)
        full_prompt = base_prompt + joined_dialogue

        # Get Prediction (Single or Best-of-Three)
        if USE_BEST_OF_THREE:
            predicted_category = get_best_of_three_prediction(service, full_prompt, llm_type)
        else:
            # --- BEGIN Rate Limiting Block (Adapt as needed) ---
            # Simple delay example - adjust based on API limits
            needs_delay = llm_type in [
                LLMType.GEMINI25, LLMType.GEMINI25_NATIVE, # Add other models needing delay
                # LLMType.DEEPSEEK_V3, LLMType.NEBIUS_DEEPSEEK_V3, LLMType.TOGETHER_DEEPSEEK_V3
            ]
            if needs_delay:
                 delay_seconds = 5 # Example delay
                 logging.debug(f"Applying delay of {delay_seconds}s for {llm_type.name}")
                 time.sleep(delay_seconds)
            # --- END Rate Limiting Block ---

            response = service.call_llm(full_prompt, llm_type, with_json_extraction=False)
            predicted_category = str(response).strip().lower()

        # Clean prediction (remove punctuation)
        predicted_category = predicted_category.translate(str.maketrans('', '', string.punctuation))

        # Validate prediction format
        if predicted_category not in ["yes", "no"]:
             logging.warning(f"LLM prediction for {file_stem} is invalid: '{predicted_category}'. Storing as 'Error'.")
             final_prediction = "Error"
        else:
             final_prediction = predicted_category

        # Return result dictionary
        return {
            'filename': file_stem,
            'actual': actual_category,
            'predicted': final_prediction
        }

    except json.JSONDecodeError as jde:
         logging.error(f"Error decoding JSON from {json_file_path}: {jde}")
         return None # Return None on error
    except Exception as e:
         logging.error(f"Error processing file {json_file_path}: {e}", exc_info=True)
         return None # Return None on error

# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":

    # --- Paths ---
    BASE_OUTPUT_DIR = Path("./abortion_classification_results") # Central output dir for CSV
    SUBTITLES_ABORTO_PATH = Path("subtitles_aborto")
    SUBTITLES_NOT_ABORTO_PATH = Path("subtitles_not_aborto")
    JSON_ABORTO_OUTPUT_DIR = SUBTITLES_ABORTO_PATH / "json_output"
    JSON_NOT_ABORTO_OUTPUT_DIR = SUBTITLES_NOT_ABORTO_PATH / "json_output"
    # Construct the full CSV path using the global filename and base dir
    CSV_OUTPUT_PATH = BASE_OUTPUT_DIR / CSV_OUTPUT_FILENAME
    # --------------------

    # --- Create Directories ---
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_ABORTO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_NOT_ABORTO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # --------------------

    # --- Process subtitles to JSON (Important Note!) ---
    # NOTE: If SRT files have been renamed, running this again without cleaning
    #       the JSON directories first might create JSON files under both old
    #       and new names. The script below will only process based on the
    #       *currently existing* JSON filenames and skip those already in the CSV,
    #       but having duplicate JSONs can be confusing.
    # Consider cleaning JSON_ABORTO_OUTPUT_DIR and JSON_NOT_ABORTO_OUTPUT_DIR
    # before running this if you have renamed SRT files.
    logging.info("Processing SRT files to JSON...")
    process_subtitles_folder_to_jsons(SUBTITLES_ABORTO_PATH, JSON_ABORTO_OUTPUT_DIR)
    process_subtitles_folder_to_jsons(SUBTITLES_NOT_ABORTO_PATH, JSON_NOT_ABORTO_OUTPUT_DIR)
    logging.info("Finished processing SRT files.")
    # ------------------------------------------------------------------------------------

    # --- Initialize LLM Service ---
    try:
        service = AIModelsService() # Ensure this class is defined and works
        logging.info(f"Initialized AI Service with model: {llm_type_var.name}")
    except NameError as ne:
         logging.error(f"LLMType or AIModelsService not defined. Please ensure imports are correct: {ne}")
         exit()
    except Exception as service_e:
         logging.error(f"Failed to initialize AIModelsService: {service_e}")
         exit()
    # -----------------------------

    # --- Load Existing Results ---
    existing_results, processed_filenames = load_existing_results(CSV_OUTPUT_PATH)
    new_results = [] # List to store results from THIS run only
    # -----------------------------

    # --- Gather Target Files based on Filename Stems ---
    target_files = {} # Dictionary to store {stem: (filepath, actual_category)}
    logging.info("Gathering target JSON files and checking for duplicates...")

    # Process 'Yes' directory
    for json_file_path in JSON_ABORTO_OUTPUT_DIR.glob("*.json"):
        stem = json_file_path.stem
        if stem in target_files:
             logging.warning(f"Duplicate stem '{stem}' found! Exists in both 'Yes' and 'No' JSON directories. Prioritizing 'Yes' category for now. File: {json_file_path}")
        target_files[stem] = (json_file_path, "Yes")

    # Process 'No' directory
    for json_file_path in JSON_NOT_ABORTO_OUTPUT_DIR.glob("*.json"):
        stem = json_file_path.stem
        if stem in target_files:
            # If already present (must have been from 'Yes' dir), log warning but don't overwrite
            if target_files[stem][1] == "Yes":
                 logging.warning(f"Duplicate stem '{stem}' found! File {json_file_path} in 'No' dir ignored as it was already found in 'Yes' dir.")
            else:
                 # This case shouldn't happen if the first loop worked correctly
                 logging.error(f"Inconsistent state for duplicate stem '{stem}'.")
        else:
            target_files[stem] = (json_file_path, "No")

    logging.info(f"Found {len(target_files)} unique JSON file stems across both directories.")
    # --------------------------------------------------

    # --- Determine files needing processing ---
    files_to_process = []
    for stem, (filepath, actual_category) in target_files.items():
        if stem not in processed_filenames:
            files_to_process.append((filepath, actual_category))

    logging.info(f"Identified {len(files_to_process)} files that need processing (not found in CSV).")
    # -----------------------------------------

    # --- Process Only New/Missing Files ---
    processed_count = 0
    for json_file_path, actual_category in files_to_process:
        processed_count += 1
        logging.info(f"--- Processing file {processed_count}/{len(files_to_process)}: {json_file_path.name} ---")
        result = process_json_file(json_file_path, actual_category, service, llm_type_var)
        if result:
            new_results.append(result)
    logging.info(f"Finished processing loop. Generated {len(new_results)} new results.")
    # ------------------------------------

    # --- Combine, Save, and Analyze Results ---
    all_results = existing_results + new_results
    logging.info(f"Total results (existing + new): {len(all_results)}")

    # Check if there are any results at all to analyze/save
    if not all_results:
        logging.warning("No results found (neither existing nor newly processed). Skipping save and analysis.")
    else:
        # Log if only existing results were loaded
        if not new_results and existing_results:
            logging.info("No new files were processed in this run; analyzing existing results from CSV.")
        elif new_results:
             logging.info(f"Combined {len(existing_results)} existing results with {len(new_results)} new results.")

        # Always save the combined (or just existing) results back to the CSV
        # This ensures the file is always up-to-date, even if no new processing occurred
        save_results_to_csv(all_results, CSV_OUTPUT_PATH)

        # Always calculate and print the confusion matrix if there are results
        calculate_and_print_confusion_matrix(all_results)

    logging.info("Script finished.")