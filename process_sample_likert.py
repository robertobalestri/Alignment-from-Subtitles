import csv
import json
import os
import logging
import re
import time
from pathlib import Path
from typing import Dict, Any, Set, List, Optional
import statistics # Already imported for median, will use for variance too

# --- Project-Specific Imports ---

try:
    from src.ai_interface.AiModels import LLMType, AIModelsService
    from src.subtitles_managing.SubtitlesManagingModels import DialogueLine # Added
    from src.subtitles_managing.subtitles_pipeline import _parse_srt_file # Added
except ImportError as e_initial:
    import sys
    current_script_path = Path(__file__).resolve()
    project_root_guess = current_script_path.parent.parent
    src_path_guess = project_root_guess / "src"

    if project_root_guess.is_dir() and str(project_root_guess) not in sys.path:
        sys.path.insert(0, str(project_root_guess))
        print(f"INFO: Aggiunto {project_root_guess} a sys.path per tentare l'importazione di src.*")
        try:
            from src.ai_interface.AiModels import LLMType, AIModelsService
            from src.subtitles_managing.SubtitlesManagingModels import DialogueLine
            from src.subtitles_managing.subtitles_pipeline import _parse_srt_file
            print("INFO: Importazione di src.* riuscita dopo aver aggiunto il project_root.")
        except ImportError as e_after_project_root:
            if src_path_guess.is_dir() and str(src_path_guess) not in sys.path:
                sys.path.insert(0, str(src_path_guess))
                print(f"INFO: Aggiunto {src_path_guess} a sys.path per tentare l'importazione diretta da 'src'.")
                try:
                    from ai_interface.AiModels import LLMType, AIModelsService
                    from subtitles_managing.SubtitlesManagingModels import DialogueLine
                    from subtitles_managing.subtitles_pipeline import _parse_srt_file
                    print("INFO: Importazione riuscita dopo aver aggiunto la directory 'src'.")
                except ImportError as e_after_src_path:
                    print(f"ERRORE: Impossibile importare dipendenze necessarie. Dettagli iniziali: {e_initial}")
                    print(f"Dettagli dopo aver aggiunto project_root ({project_root_guess}): {e_after_project_root}")
                    print(f"Dettagli dopo aver aggiunto src_path ({src_path_guess}): {e_after_src_path}")
                    print("Assicurati che la directory 'src' sia nel PYTHONPATH o che lo script sia eseguito dalla directory corretta.")
                    exit(1)
            else:
                print(f"ERRORE: Impossibile importare dipendenze necessarie. Dettagli: {e_initial}")
                print(f"Tentativo con project_root ({project_root_guess}) fallito: {e_after_project_root}")
                print("Assicurati che la directory 'src' sia nel PYTHONPATH o che lo script sia eseguito dalla directory corretta.")
                exit(1)
    else:
        print(f"ERRORE: Impossibile importare dipendenze necessarie. Dettagli: {e_initial}")
        print("Assicurati che la directory 'src' sia nel PYTHONPATH o che lo script sia eseguito dalla directory corretta.")
        exit(1)


# --- End Imports ---

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(lineno)d: %(message)s')
logger = logging.getLogger(__name__)
# -----------------------------

# --- Configuration Variables ---
# Make sure these paths are correct for your setup
SRT_SOURCE_DIR = Path("subtitles_campione_razzismo_IT_per_likert")
SUBTITLES_JSON_INPUT_DIR = Path("subtitles_campione_razzismo_IT_per_likert/json_output")

SUBTITLES_TEXT_OUTPUT_DIR = SUBTITLES_JSON_INPUT_DIR.parent / "text_output"

LLM_TYPE_TO_USE = LLMType.GPT4_1 # Or GPT4O, GEMINI15, etc.
OUTPUT_CSV_PATH = Path("output_test_su_campioni/gpt41_likert_episodio_razzismo_campione_IT.csv")
PROMPT_FILE_PATH = Path("prompts") / "racism_ITA_likert_evaluation.txt"

NUMBER_OF_ATTEMPTS = 10 # Number of LLM calls per episode

# --- Load Base Prompt from file ---
try:
    with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as f:
        base_prompt = f.read()
    logger.info(f"Prompt caricato da: {PROMPT_FILE_PATH}")
except FileNotFoundError:
    logger.error(f"File del prompt non trovato: {PROMPT_FILE_PATH}")
    # Decide how critical this is. Exit or use a fallback.
    # base_prompt = "FALLBACK_PROMPT_ERROR_LOADING_FILE"
    exit(1)
except Exception as e:
    logger.error(f"Errore nel caricamento del prompt da {PROMPT_FILE_PATH}: {e}")
    # base_prompt = "FALLBACK_PROMPT_ERROR_LOADING_FILE"
    exit(1)
# --- End Prompt Loading ---

# --- End Configuration ---


# ==============================================================================
# FUNZIONE PER CONVERTIRE SRT IN JSON SE MANCANTI
# ==============================================================================
def convert_srts_to_json_if_missing(srt_source_dir: Path, json_output_dir: Path):
    """
    Controlla la srt_source_dir per file .srt.
    Per ogni .srt, verifica se un .json corrispondente esiste in json_output_dir.
    Se il .json non esiste, converte l'.srt e lo salva in json_output_dir.
    I file JSON vengono creati direttamente in json_output_dir (struttura piatta).
    """
    logger.info(f"Avvio controllo e conversione SRT da '{srt_source_dir}' a JSON in '{json_output_dir}'...")
    srt_files_found = 0
    json_files_created = 0
    json_already_exist = 0
    conversion_errors = 0
    conversion_error_files = [] # List to store paths of failed SRT files

    if not srt_source_dir.is_dir():
        logger.warning(f"La directory sorgente SRT '{srt_source_dir}' non esiste o non è una directory. Salto la conversione SRT.")
        return

    # Assicurati che la directory di output JSON esista
    json_output_dir.mkdir(parents=True, exist_ok=True)

    for srt_path in srt_source_dir.rglob("*.srt"): # Cerca ricorsivamente gli SRT
        srt_files_found += 1
        # Usa lo stem del file SRT originale per il nome JSON
        json_file_name = srt_path.stem + ".json"
        json_path = json_output_dir / json_file_name

        if json_path.exists():
            logger.debug(f"Il file JSON '{json_path.name}' esiste già in '{json_output_dir}'. Salto la conversione per '{srt_path.name}'.")
            json_already_exist += 1
            continue

        logger.info(f"Conversione di '{srt_path.name}' in '{json_path.name}' nella directory '{json_output_dir}'...")
        try:
            # Usa la funzione di parsing importata (potrebbe richiedere aggiustamenti se la firma è diversa)
            dialogue_entries: List[DialogueLine] = _parse_srt_file(srt_path) # Passa il path come stringa se richiesto
            if not dialogue_entries:
                logger.warning(f"Nessun dato analizzato da '{srt_path.name}', impossibile creare JSON.")
                conversion_errors += 1
                if srt_path not in conversion_error_files:
                    conversion_error_files.append(srt_path)
                continue

            # Assumi che DialogueLine abbia un metodo to_dict() o simile
            json_data = [entry.to_dict() for entry in dialogue_entries]

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Creato JSON: '{json_path.name}' in '{json_output_dir}'.")
            json_files_created += 1

        except Exception as e:
            logger.error(f"Errore durante la conversione di '{srt_path.name}' in '{json_path.name}': {e}", exc_info=True)
            conversion_errors += 1
            if srt_path not in conversion_error_files:
                conversion_error_files.append(srt_path)
            if json_path.exists():
                try:
                    json_path.unlink()
                    logger.info(f"Rimosso file JSON parziale '{json_path.name}' a seguito di errore.")
                except OSError as ose:
                    logger.error(f"Errore durante la rimozione del file JSON parziale '{json_path.name}': {ose}")

    logger.info("--- Riepilogo Conversione SRT in JSON ---")
    logger.info(f"  Directory Sorgente SRT: {srt_source_dir}")
    logger.info(f"  Directory Destinazione JSON: {json_output_dir}")
    logger.info(f"  File SRT Trovati: {srt_files_found}")
    logger.info(f"  File JSON Creati: {json_files_created}")
    logger.info(f"  File JSON Già Esistenti (saltati): {json_already_exist}")
    logger.info(f"  Errori di Conversione: {conversion_errors}")

    if conversion_error_files:
        logger.warning("--- File SRT che hanno fallito la conversione ---")
        conversion_error_files.sort()
        for failed_path in conversion_error_files:
            # Usa os.path.normpath per un output più pulito
            logger.warning(f"  - {os.path.normpath(failed_path)}")
    elif conversion_errors == 0 and srt_files_found > 0 and json_files_created > 0:
         logger.info("Conversione da SRT a JSON completata con successo per i file mancanti.")
    elif conversion_errors == 0 and srt_files_found > 0 and json_files_created == 0 and json_already_exist == srt_files_found:
        logger.info("Nessun nuovo file JSON creato, tutti i corrispondenti JSON esistevano già.")
    elif srt_files_found == 0:
        logger.info(f"Nessun file SRT trovato in '{srt_source_dir}'. Nessuna conversione eseguita.")
    else:
        logger.info("Controllo e conversione SRT completati.")

# ==============================================================================


def parse_episode_code(filename_stem: str) -> Optional[Dict[str, str]]:
    """
    Analizza un episode_code (es. 'CMS02E09', 'La dottoressa GioS01E01',
    'Stargate SG1 S05E10') per estrarre serie, stagione ed episodio.
    Cattura tutto fino all'ULTIMO pattern SxxExx alla fine della stringa.
    Restituisce un dizionario con 'series', 'season', 'episode' o None se il parsing fallisce.
    """
    # Regex Modificato per GREEDY match della serie:
    # ^          -> Inizio stringa
    # (.*)       -> GRUPPO 1 (Serie): Qualsiasi carattere (.), zero o più (*),
    #               il PIÙ possibile (greedy, default), fino a permettere
    #               al resto del pattern di matchare alla fine.
    # S          -> 'S' letterale (ignorando maiuscole/minuscole)
    # (\d{2})    -> GRUPPO 2 (Stagione): Due cifre
    # E          -> 'E' letterale (ignorando maiuscole/minuscole)
    # (\d{2})    -> GRUPPO 3 (Episodio): Due cifre
    # $          -> Fine stringa
    match = re.match(r"^(.*)S(\d{2})E(\d{2})$", filename_stem, re.IGNORECASE) # <--- USE GREEDY (.*)

    if match:
        # Pulisci eventuali spazi extra dal nome della serie catturato
        # e metti in maiuscolo per coerenza (o rimuovi .upper() se preferisci)
        series = match.group(1).strip().upper()
        season = match.group(2)
        episode = match.group(3)
        logger.debug(f"Parsing riuscito (Regola Greedy SxxExx Finale) per '{filename_stem}': Serie='{series}', Stagione='{season}', Episodio='{episode}'")
        return {"series": series, "season": season, "episode": episode}
    else:
        # Aggiorna il messaggio di warning
        logger.warning(f"Parsing fallito per '{filename_stem}'. Formato non riconosciuto (atteso Nome_Serie_Qualsiasi terminante in SxxExx).")
        return None

def load_subtitles_from_json_file(json_file_path: Path) -> str:
    """
    Carica il testo completo dei sottotitoli da un file JSON specificato.
    Il file JSON dovrebere contenere una lista di dizionari, ognuno con una chiave 'text'.
    """
    logger.debug(f"Tentativo di caricare i sottotitoli da: {json_file_path}")
    try:
        if not json_file_path.is_file():
            error_msg = f"[Errore: File JSON dei sottotitoli non trovato a {json_file_path}]"
            logger.error(error_msg)
            return error_msg

        with open(json_file_path, 'r', encoding='utf-8') as f:
            subtitles_data = json.load(f)

        if not isinstance(subtitles_data, list):
            error_msg = f"[Errore: formato JSON dei sottotitoli non valido (non è una lista) in {json_file_path}]"
            logger.error(error_msg)
            return error_msg

        full_text_parts = []
        expected_keys = ['text', 'content'] # Keys to check for subtitle text
        for item in subtitles_data:
            if isinstance(item, dict):
                found_key = None
                for key in expected_keys:
                    if key in item:
                        full_text_parts.append(str(item[key]))
                        found_key = key
                        break # Found the text for this item
                if found_key is None:
                     logger.warning(f"Elemento ignorato in {json_file_path} perché non contiene chiavi valide ({expected_keys}): {item}")
            else:
                logger.warning(f"Elemento ignorato in {json_file_path} perché non è un dizionario: {item}")


        if not full_text_parts and subtitles_data:
             logger.warning(f"Nessun testo di sottotitoli valido trovato in {json_file_path} nonostante il file esista e contenga una lista.")

        concatenated_text = "\n".join(full_text_parts)
        logger.debug(f"Sottotitoli caricati con successo da {json_file_path}. Lunghezza testo: {len(concatenated_text)} caratteri.")
        return concatenated_text

    except json.JSONDecodeError as e:
        error_msg = f"[Errore: JSONDecodeError nel file dei sottotitoli - {e} in {json_file_path}]"
        logger.error(error_msg)
        return error_msg
    except FileNotFoundError: # Should be caught by is_file() but good practice
        error_msg = f"[Errore: File JSON dei sottotitoli non trovato (eccezione imprevista) a {json_file_path}]"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"[Errore: Eccezione imprevista '{type(e).__name__}' durante il caricamento dei sottotitoli da {json_file_path} - {e}]"
        logger.error(error_msg, exc_info=True)
        return error_msg

def call_llm(prompt_text: str, service: AIModelsService, llm_type: LLMType, episode_code_for_logging: str = "N/A") -> Dict[str, Any]:
    logger.info(f"Chiamata LLM per episode_code: {episode_code_for_logging} con tipo LLM: {llm_type.name}")
    max_retries = 3
    retry_delay_seconds = 10
    default_error_response = {
        "score": "LLM_CALL_ERROR",
        "explanation": "Errore nell'elaborazione della risposta LLM dopo i tentativi.",
        "why_not_lower": "",
        "why_not_higher": ""
    }

    for attempt in range(max_retries):
        try:
            # Assuming call_llm returns the raw string or potentially a dict if pre-parsed
            raw_response = service.call_llm(prompt_text, llm_type, with_json_extraction=True)
            logger.debug(f"Risposta grezza dall'LLM (tentativo {attempt + 1}) per {episode_code_for_logging}: {raw_response}")

            response_data = None
            cleaned_response_str = None

            if isinstance(raw_response, dict):
                 # Already parsed by the service
                 response_data = raw_response
            elif isinstance(raw_response, str):
                # Clean up potential markdown code fences
                cleaned_response_str = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_response.strip(), flags=re.MULTILINE | re.IGNORECASE)
                if not cleaned_response_str:
                    logger.warning(f"LLM ha restituito una stringa vuota dopo la pulizia per {episode_code_for_logging}. Tentativo {attempt + 1}/{max_retries}.")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay_seconds)
                        continue
                    else:
                        default_error_response["explanation"] = "LLM ha restituito una stringa vuota dopo la pulizia e tutti i tentativi."
                        return default_error_response
                try:
                    response_data = json.loads(cleaned_response_str)
                except json.JSONDecodeError as json_e:
                    logger.warning(f"Impossibile fare il parsing della risposta JSON dall'LLM per {episode_code_for_logging}: {json_e}. Tentativo {attempt + 1}/{max_retries}. Risposta pulita: '{cleaned_response_str}'")
                    if attempt == max_retries - 1:
                        default_error_response["explanation"] = f"Impossibile fare il parsing della risposta JSON dall'LLM dopo i tentativi: {json_e}. Risposta pulita: '{cleaned_response_str}', Grezza: '{raw_response}'"
                        return default_error_response
                    else:
                        time.sleep(retry_delay_seconds)
                        continue # Go to next retry
            else:
                 # Handle unexpected response types
                 logger.warning(f"La risposta LLM è di tipo inatteso: {type(raw_response)} per {episode_code_for_logging}. Tentativo {attempt + 1}/{max_retries}. Contenuto: {raw_response}")
                 if attempt == max_retries - 1:
                     default_error_response["explanation"] = f"La risposta LLM è di tipo inatteso ({type(raw_response)}) dopo i tentativi."
                     return default_error_response
                 else:
                    time.sleep(retry_delay_seconds)
                    continue # Go to next retry

            # Validate the structure of the parsed/received dictionary
            if response_data and isinstance(response_data, dict) and all(key in response_data for key in ["score", "explanation"]):
                # Ensure optional keys exist
                response_data.setdefault("why_not_lower", "")
                response_data.setdefault("why_not_higher", "")
                logger.info(f"Risposta LLM valida ricevuta per {episode_code_for_logging} (tentativo {attempt + 1}).")
                return response_data
            else:
                logger.warning(f"LLM ha restituito una struttura dati incompleta o non valida per {episode_code_for_logging}: {response_data}. Tentativo {attempt + 1}/{max_retries}.")
                if attempt == max_retries - 1:
                    default_error_response["explanation"] = f"LLM ha restituito una struttura dati incompleta o non valida dopo i tentativi: {response_data}"
                    return default_error_response
                else:
                    time.sleep(retry_delay_seconds)
                    # No need to continue here, the loop will naturally proceed to the next attempt

        except Exception as e:
            logger.error(f"Errore imprevisto durante la chiamata LLM o l'elaborazione della risposta per {episode_code_for_logging} (tentativo {attempt + 1}): {e}", exc_info=True)
            if attempt == max_retries - 1:
                default_error_response["explanation"] = f"Errore imprevisto durante la chiamata LLM o l'elaborazione della risposta dopo i tentativi: {e}."
                return default_error_response
            else:
                time.sleep(retry_delay_seconds)

    # This point should ideally not be reached if default_error_response is always returned on failure
    logger.error(f"Chiamata LLM fallita per {episode_code_for_logging} dopo {max_retries} tentativi (flusso di controllo imprevisto).")
    return default_error_response


def load_existing_results_from_csv(csv_path: Path) -> tuple[List[Dict[str, Any]], Set[tuple[str, int]]]:
    existing_results: List[Dict[str, Any]] = []
    processed_episode_attempts: Set[tuple[str, int]] = set()
    # Fieldnames should match the expected CSV structure
    fieldnames = ['series', 'season', 'episode', 'episode_code', 'attempt_number', 'score', 'explanation', 'why_not_lower', 'why_not_higher']

    if not csv_path.is_file():
        logger.info(f"File CSV dei risultati non trovato a {csv_path}. Si inizia da zero.")
        return existing_results, processed_episode_attempts

    logger.info(f"Caricamento dei risultati esistenti da: {csv_path}")
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # More robust header check
            if not reader.fieldnames:
                logger.error(f"Il file CSV {csv_path} è vuoto o non contiene header. Impossibile caricare i risultati esistenti. Si inizia da zero.")
                return [], set()

            # Check if all expected fieldnames are present
            missing_headers = [f for f in fieldnames if f not in reader.fieldnames]
            extra_headers = [f for f in reader.fieldnames if f not in fieldnames] # Optional: check for unexpected columns

            if missing_headers:
                 error_msg = (f"Il file CSV {csv_path} ha header mancanti. "
                              f"Mancanti: {missing_headers}. Attesi: {fieldnames}. "
                              f"Trovati: {reader.fieldnames}. "
                              f"Impossibile caricare i risultati esistenti. Si inizia da zero.")
                 logger.error(error_msg)
                 return [], set()
            if extra_headers:
                logger.warning(f"Il file CSV {csv_path} contiene colonne extra non attese: {extra_headers}. Saranno ignorate durante il caricamento.")


            for i, row in enumerate(reader):
                # Check if row is a dictionary and has the essential keys
                if not isinstance(row, dict):
                     logger.warning(f"Riga {i+2} in {csv_path} non è un dizionario valido, saltata: {row}")
                     continue

                episode_code = row.get('episode_code')
                attempt_str = row.get('attempt_number')

                # Ensure all required fields are present, even if potentially empty
                valid_row = True
                for field in fieldnames:
                    if field not in row:
                         logger.warning(f"Riga {i+2} in {csv_path} manca della colonna '{field}', saltata: {row}")
                         valid_row = False
                         break
                if not valid_row:
                    continue


                if episode_code and attempt_str:
                    try:
                        attempt_number = int(attempt_str)
                        # Add the row as is (it's already a dict)
                        existing_results.append(row)
                        processed_episode_attempts.add((episode_code, attempt_number))
                    except ValueError:
                        logger.warning(f"Riga {i+2} saltata per 'attempt_number' non numerico in {csv_path}: {row}")
                    except TypeError:
                        logger.warning(f"Riga {i+2} saltata per 'attempt_number' di tipo non valido in {csv_path}: {row}")
                else:
                    # Log missing essential identifiers
                    missing_id = "'episode_code'" if not episode_code else "'attempt_number'"
                    logger.warning(f"Riga {i+2} saltata per {missing_id} mancante o vuoto in {csv_path}: {row}")

        logger.info(f"Caricati {len(existing_results)} risultati esistenti. Trovati {len(processed_episode_attempts)} tentativi di episodi processati.")
    except FileNotFoundError:
        logger.info(f"File CSV non trovato a {csv_path}. Si inizia da zero.") # Should be handled by is_file() check
    except Exception as e:
        logger.error(f"Errore imprevisto durante la lettura del file CSV {csv_path}: {e}. Si procede senza risultati esistenti.", exc_info=True)
        return [], set() # Reset to safe state
    return existing_results, processed_episode_attempts


def save_results_to_csv(results_data: List[Dict[str, Any]], output_csv_path: Path):
    if not results_data:
        logger.warning("Nessun dato di risultato (nuovo o vecchio) fornito per il salvataggio su CSV.")
        return

    # Fieldnames must match the structure of dictionaries in results_data
    fieldnames = ['series', 'season', 'episode', 'episode_code', 'attempt_number', 'score', 'explanation', 'why_not_lower', 'why_not_higher']
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Salvataggio di {len(results_data)} risultati totali su CSV: {output_csv_path}")
        # Filter out any potential non-dictionary items just in case
        valid_results = [r for r in results_data if isinstance(r, dict)]
        if len(valid_results) < len(results_data):
             logger.warning(f"Trovate {len(results_data) - len(valid_results)} voci non-dizionario nei dati dei risultati. Salvataggio solo di quelle valide.")

        # Ensure all rows have all expected keys, adding missing ones with None or empty string if needed
        processed_results = []
        for row in valid_results:
            processed_row = {key: row.get(key, '') for key in fieldnames} # Use '' as default for missing keys
            processed_results.append(processed_row)


        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(processed_results)
        logger.info(f"Risultati salvati con successo su {output_csv_path}")
    except IOError as e:
        logger.error(f"Errore di I/O durante la scrittura del file CSV {output_csv_path}: {e}")
    except Exception as e:
        logger.error(f"Errore imprevisto durante la scrittura del CSV: {e}", exc_info=True)


def main():
    # Ensure output directories exist
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    # --- NEW ---
    # Ensure the text output directory exists
    SUBTITLES_TEXT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"I file di testo dei dialoghi estratti verranno salvati in: {SUBTITLES_TEXT_OUTPUT_DIR}")
    # -----------

    # Convert SRTs to JSON if needed before starting main processing
    logger.info(f"Controllo e conversione dei file SRT da '{SRT_SOURCE_DIR}' a JSON in '{SUBTITLES_JSON_INPUT_DIR}' se mancanti.")
    convert_srts_to_json_if_missing(SRT_SOURCE_DIR, SUBTITLES_JSON_INPUT_DIR)
    logger.info("Controllo/Conversione SRT completati. Avvio elaborazione principale.")

    try:
        service = AIModelsService()
        logger.info(f"AIModelsService inizializzato. Modello da usare: {LLM_TYPE_TO_USE.name}")
    except Exception as e:
        logger.error(f"Impossibile inizializzare AIModelsService: {e}", exc_info=True)
        return # Cannot proceed without the service

    # Load existing results to avoid re-processing
    all_results_data, processed_episode_attempts = load_existing_results_from_csv(OUTPUT_CSV_PATH)
    logger.info(f"Caricati {len(all_results_data)} risultati esistenti e {len(processed_episode_attempts)} tentativi di episodi già processati.")

    logger.info(f"Scansione dei file JSON in: {SUBTITLES_JSON_INPUT_DIR}")
    json_files_to_process = sorted(list(SUBTITLES_JSON_INPUT_DIR.glob("*.json"))) # Sort for consistent order
    logger.info(f"Trovati {len(json_files_to_process)} file JSON totali.")

    new_results_count = 0
    for json_file_path in json_files_to_process:
        episode_code_from_filename = json_file_path.stem # e.g., 'SERIESNAME S01E01'
        # Normalize the episode code if necessary (e.g., remove trailing dots)
        normalized_episode_code = episode_code_from_filename.rstrip('.')

        logger.info(f"--- Elaborazione di: {json_file_path.name} (Codice Episodio Normalizzato: {normalized_episode_code}) ---")

        parsed_info = parse_episode_code(normalized_episode_code)
        if not parsed_info:
            logger.error(f"Impossibile analizzare le informazioni dal nome file normalizzato '{normalized_episode_code}' (originale: {json_file_path.name}). Episodio saltato.")
            # Add error rows for all attempts of this unparsable file if they haven't been added already
            needs_save = False
            for attempt_n in range(1, NUMBER_OF_ATTEMPTS + 1):
                 if (normalized_episode_code, attempt_n) not in processed_episode_attempts:
                     error_row = {
                         'series': 'PARSE_ERROR', 'season': 'N/A', 'episode': 'N/A',
                         'episode_code': normalized_episode_code, 'attempt_number': attempt_n,
                         'score': 'FILENAME_PARSE_ERROR',
                         'explanation': f'Impossibile analizzare il nome del file: {json_file_path.name}',
                         'why_not_lower': '', 'why_not_higher': ''
                     }
                     all_results_data.append(error_row)
                     processed_episode_attempts.add((normalized_episode_code, attempt_n))
                     needs_save = True
            if needs_save:
                save_results_to_csv(all_results_data, OUTPUT_CSV_PATH) # Save the errors
            continue # Skip to the next file

        series_name = parsed_info['series']
        season_num = parsed_info['season']
        episode_num = parsed_info['episode']

        subtitles_text = load_subtitles_from_json_file(json_file_path)

        # Check if loading failed or text is empty/whitespace
        if subtitles_text.startswith("[Errore:") or not subtitles_text.strip():
            log_detail = subtitles_text if subtitles_text.startswith("[Errore:") else "Testo sottotitoli vuoto o mancante."
            logger.error(f"Impossibile caricare o testo vuoto per i sottotitoli da {json_file_path.name} (Codice: {normalized_episode_code}). Dettaglio: {log_detail}. Saltato.")
            # Add error rows for all attempts of this file if subtitle loading failed
            needs_save = False
            for attempt_n in range(1, NUMBER_OF_ATTEMPTS + 1):
                 if (normalized_episode_code, attempt_n) not in processed_episode_attempts:
                    error_row = {
                        'series': series_name, 'season': season_num, 'episode': episode_num,
                        'episode_code': normalized_episode_code, 'attempt_number': attempt_n,
                        'score': 'SUBTITLE_LOAD_ERROR',
                        'explanation': log_detail,
                        'why_not_lower': '', 'why_not_higher': ''
                    }
                    all_results_data.append(error_row)
                    processed_episode_attempts.add((normalized_episode_code, attempt_n))
                    needs_save = True
            if needs_save:
                save_results_to_csv(all_results_data, OUTPUT_CSV_PATH)
            continue # Skip to the next file

        # --- NEW: Save extracted text to .txt file ---
        txt_output_filename = normalized_episode_code + ".txt"
        txt_output_path = SUBTITLES_TEXT_OUTPUT_DIR / txt_output_filename
        try:
            with open(txt_output_path, 'w', encoding='utf-8') as f_txt:
                f_txt.write(subtitles_text)
            logger.info(f"Testo dei sottotitoli salvato con successo in: {txt_output_path}")
        except IOError as e:
            logger.error(f"Impossibile salvare il file di testo {txt_output_path}: {e}")
        except Exception as e:
            logger.error(f"Errore imprevisto durante il salvataggio del file di testo {txt_output_path}: {e}", exc_info=True)
        # --- END NEW ---


        full_prompt = base_prompt.replace("{subtitles_text}", subtitles_text)

        episode_needs_save = False
        for attempt_number in range(1, NUMBER_OF_ATTEMPTS + 1):
            attempt_id_tuple = (normalized_episode_code, attempt_number)
            attempt_id_for_log = f"{normalized_episode_code}-Attempt{attempt_number}"

            if attempt_id_tuple in processed_episode_attempts:
                logger.info(f"Tentativo {attempt_id_for_log} già processato. Saltato.")
                continue

            logger.info(f"--> Chiamata LLM per {attempt_id_for_log}...")
            start_time = time.time()
            llm_response = call_llm(full_prompt, service, LLM_TYPE_TO_USE, attempt_id_for_log)
            end_time = time.time()
            logger.info(f"<-- Risposta LLM ricevuta per {attempt_id_for_log} in {end_time - start_time:.2f} secondi.")

            result_row = {
                'series': series_name,
                'season': season_num,
                'episode': episode_num,
                'episode_code': normalized_episode_code,
                'attempt_number': attempt_number,
                'score': llm_response.get('score', 'LLM_RESPONSE_KEY_ERROR'),
                'explanation': llm_response.get('explanation', 'LLM_RESPONSE_KEY_ERROR'),
                'why_not_lower': llm_response.get('why_not_lower', 'LLM_RESPONSE_KEY_ERROR'),
                'why_not_higher': llm_response.get('why_not_higher', 'LLM_RESPONSE_KEY_ERROR')
            }
            all_results_data.append(result_row)
            processed_episode_attempts.add(attempt_id_tuple)
            new_results_count += 1
            episode_needs_save = True # Mark that we need to save for this episode

            # Optional: Add a small delay between attempts
            # time.sleep(1)

        if episode_needs_save:
            save_results_to_csv(all_results_data, OUTPUT_CSV_PATH)
            logger.info(f"Risultati aggiornati salvati per {normalized_episode_code}. Totale risultati finora: {len(all_results_data)}.")
        else:
             logger.info(f"Nessun nuovo tentativo processato per {normalized_episode_code}. Nessun salvataggio necessario.")

    logger.info(f"Elaborazione completata. {new_results_count} nuovi tentativi di episodi processati.")
    logger.info(f"Totale risultati nel CSV: {len(all_results_data)}.")
    logger.info(f"I risultati finali sono stati salvati in: {OUTPUT_CSV_PATH}")
    logger.info(f"I file di testo dei dialoghi estratti sono stati salvati in: {SUBTITLES_TEXT_OUTPUT_DIR}") # Added final confirmation

    calculate_and_print_median_variance(all_results_data)


def calculate_and_print_median_variance(results_data: List[Dict[str, Any]]):
    """
    Calculates and prints the median and variance of scores for each episode.
    Handles cases with fewer than 2 scores for variance calculation.
    """
    logger.info("\n--- Calcolo dei Punteggi Mediani e Varianza per Episodio ---")
    if not results_data:
        logger.info("Nessun dato disponibile per calcolare i punteggi mediani e la varianza.")
        return

    episodes_scores: Dict[str, List[float]] = {}
    for row in results_data:
        # Ensure row is a dictionary and has the required keys
        if not isinstance(row, dict):
            logger.warning(f"Elemento non dizionario nei risultati, ignorato per le statistiche: {row}")
            continue
        episode_code = row.get('episode_code')
        score_str = row.get('score')

        if episode_code and score_str:
            try:
                score = float(score_str) # Convert score to float
                if episode_code not in episodes_scores:
                    episodes_scores[episode_code] = []
                episodes_scores[episode_code].append(score)
            except (ValueError, TypeError):
                # Ignore scores that cannot be converted to float
                logger.debug(f"Punteggio non numerico '{score_str}' per l'episodio {episode_code}, ignorato per le statistiche.")
                pass # Logged as debug, continue processing other scores/rows
        #else: # Optional: Log if essential keys are missing, though load/save should handle this
        #    logger.warning(f"Riga con 'episode_code' o 'score' mancante, ignorata per le statistiche: {row}")


    if not episodes_scores:
        logger.info("Nessun punteggio numerico valido trovato per calcolare mediane e varianze.")
        return

    print("\n--- Punteggi Mediani e Varianza per Episodio ---")
    # Sort episodes for consistent output order
    sorted_episode_codes = sorted(episodes_scores.keys())

    for episode_code in sorted_episode_codes:
        scores = episodes_scores[episode_code]
        if scores: # Check if there are any valid scores for this episode
            median_score = statistics.median(scores)
            variance_score_str = "N/A" # Default if variance cannot be calculated

            if len(scores) >= 2:
                try:
                    variance_score = statistics.variance(scores)
                    variance_score_str = f"{variance_score:.2f}" # Format variance to 2 decimal places
                except statistics.StatisticsError as e:
                    # This shouldn't happen with the len check, but catch just in case
                    logger.error(f"Errore nel calcolo della varianza per {episode_code} con {len(scores)} punteggi: {scores}. Errore: {e}")
                    variance_score_str = "Calc Error"
            elif len(scores) == 1:
                 variance_score_str = "N/A (1 score)" # More specific N/A reason

            # Print results including variance
            print(f"Episodio: {episode_code}, Punteggi: {scores}, Mediana: {median_score:.2f}, Varianza: {variance_score_str}")
        else:
            # This case should theoretically not happen if episode_code is in the dict,
            # but included for completeness.
            print(f"Episodio: {episode_code}, Nessun punteggio numerico valido trovato.")

    logger.info("Calcolo dei punteggi mediani e delle varianze completato.")


if __name__ == "__main__":
    main()