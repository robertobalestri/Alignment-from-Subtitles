import csv
import json
import os
import logging
import re
import time
from pathlib import Path
from typing import Dict, Any, Set


# --- Project-Specific Imports ---

try:
    from src.ai_interface.AiModels import LLMType, AIModelsService
except ImportError as e_initial:
    import sys
    current_script_path = Path(__file__).resolve()
    project_root_guess = current_script_path.parent.parent # Assuming script is in a subfolder like 'scripts'
    src_path_guess = project_root_guess / "src"
    
    # Try adding project_root_guess (e.g., /path/to/your_project)
    if project_root_guess.is_dir() and str(project_root_guess) not in sys.path:
        sys.path.insert(0, str(project_root_guess))
        print(f"INFO: Aggiunto {project_root_guess} a sys.path per tentare l'importazione di src.*")
        try:
            from src.ai_interface.AiModels import LLMType, AIModelsService
            print("INFO: Importazione di src.ai_interface.AiModels riuscita dopo aver aggiunto il project_root.")
        except ImportError as e_after_project_root:
            # If that failed, try adding src_path_guess (e.g., /path/to/your_project/src)
            # This is less common if 'src' is a package, but can happen.
            if src_path_guess.is_dir() and str(src_path_guess) not in sys.path:
                sys.path.insert(0, str(src_path_guess)) # Try adding src itself
                print(f"INFO: Aggiunto {src_path_guess} a sys.path per tentare l'importazione diretta da 'src'.")
                try:
                    # This import style might change depending on how AiModels is structured within src
                    from ai_interface.AiModels import LLMType, AIModelsService 
                    print("INFO: Importazione di ai_interface.AiModels riuscita dopo aver aggiunto la directory 'src'.")
                except ImportError as e_after_src_path:
                    print(f"ERRORE: Impossibile importare AIModelsService e LLMType. Dettagli iniziali: {e_initial}")
                    print(f"Dettagli dopo aver aggiunto project_root ({project_root_guess}): {e_after_project_root}")
                    print(f"Dettagli dopo aver aggiunto src_path ({src_path_guess}): {e_after_src_path}")
                    print("Assicurati che la directory 'src' sia nel PYTHONPATH, che lo script sia eseguito dalla directory corretta, o che la struttura del progetto sia come previsto.")
                    exit(1)
            else:
                print(f"ERRORE: Impossibile importare AIModelsService e LLMType. Dettagli: {e_initial}")
                print(f"Tentativo con project_root ({project_root_guess}) fallito: {e_after_project_root}")
                print("Assicurati che la directory 'src' sia nel PYTHONPATH, che lo script sia eseguito dalla directory corretta, o che la struttura del progetto sia come previsto.")
                exit(1)
    else: # project_root_guess was not a dir or already in path, initial error stands
        print(f"ERRORE: Impossibile importare AIModelsService e LLMType. Dettagli: {e_initial}")
        print("Assicurati che la directory 'src' sia nel PYTHONPATH, che lo script sia eseguito dalla directory corretta, o che la struttura del progetto sia come previsto.")
        exit(1)


# --- End Imports ---

# --- Logging Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(lineno)d: %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------

# --- Configuration Variables ---

INPUT_CSV_PATH = "output_experiment_gpt4_1_IT/gpt_4_1_racism_classification_results_IT.csv"
SUBTITLES_DATASET_TYPE = "IT" # Usato solo per costruire il default di SUBTITLES_JSON_BASE_PATH
SUBTITLES_JSON_BASE_PATH = f"subtitles_dataset_{SUBTITLES_DATASET_TYPE}_renamed_structured"
LLM_TYPE_TO_USE = LLMType.GPT4_1
OUTPUT_FILENAME_TEMPLATE = "IT_racism_gpt_4_1_llm_episode_scores.csv" # Aggiornato nome file

# Il percorso del file prompt è relativo alla directory di lavoro corrente (CWD)
PROMPT_FILE_PATH = Path("prompts") / "racism_evaluation_likert_episode_prompt.txt"

# --- End Configuration ---

def load_subtitles_text(series_name: str, season_num_str: str, episode_num_str: str, base_path_str: str, episode_code_for_log: str) -> str:
    """
    Carica il testo completo dei sottotitoli per un dato episodio usando series_name, season_num, episode_num.
    Il file JSON dovrebbe contenere una lista di dizionari, ognuno con una chiave 'text'.
    La struttura del percorso attesa è: {base_path}/{series_name}/S{ss}/E{ee}/{series_name}S{ss}E{ee}.json
    """
    logger.debug(f"Tentativo di caricare i sottotitoli per Serie: '{series_name}', Stagione: {season_num_str}, Episodio: {episode_num_str} (Episode Code CSV: {episode_code_for_log})")
    try:
        if not all([series_name, season_num_str, episode_num_str]):
            error_msg = f"[Errore: Valori mancanti per series_name, season_num_str o episode_num_str. Ricevuto: S='{series_name}', Se='{season_num_str}', Ep='{episode_num_str}']"
            logger.error(error_msg + f" (per episode_code: {episode_code_for_log})")
            return error_msg

        season_padded = season_num_str.zfill(2)
        episode_padded = episode_num_str.zfill(2)

        season_folder_name = f"S{season_padded}"
        episode_folder_name = f"E{episode_padded}"
        
        filename = f"{series_name}{season_folder_name}{episode_folder_name}.json"
        
        subtitles_file_path = Path(base_path_str) / series_name / season_folder_name / episode_folder_name / filename

        logger.info(f"Percorso costruito per il file JSON dei sottotitoli: {subtitles_file_path} (per episode_code: {episode_code_for_log})")

        if not subtitles_file_path.is_file():
            error_msg = f"[Errore: File JSON dei sottotitoli non trovato a {subtitles_file_path}]"
            logger.error(error_msg + f" (per episode_code: {episode_code_for_log})")
            return error_msg

        with open(subtitles_file_path, 'r', encoding='utf-8') as f:
            subtitles_data = json.load(f)
        
        if not isinstance(subtitles_data, list):
            error_msg = f"[Errore: formato JSON dei sottotitoli non valido (non è una lista) in {subtitles_file_path}]"
            logger.error(error_msg + f" (per episode_code: {episode_code_for_log})")
            return error_msg

        full_text = [str(item['text']) for item in subtitles_data if isinstance(item, dict) and 'text' in item]
        if not full_text and subtitles_data: 
             logger.warning(f"Nessun testo di sottotitoli valido trovato in {subtitles_file_path} nonostante il file esista e contenga una lista (per episode_code: {episode_code_for_log}).")

        concatenated_text = "\n".join(full_text)
        logger.debug(f"Sottotitoli caricati con successo per Serie: '{series_name}', Stagione: {season_padded}, Episodio: {episode_padded}. Lunghezza testo: {len(concatenated_text)} caratteri (per episode_code: {episode_code_for_log}).")
        return concatenated_text

    except json.JSONDecodeError as e:
        error_msg = f"[Errore: JSONDecodeError nel file dei sottotitoli - {e} in {str(subtitles_file_path) if 'subtitles_file_path' in locals() else 'PATH_NON_DEFINITO'}]"
        logger.error(error_msg + f" (per episode_code: {episode_code_for_log})")
        return error_msg
    except FileNotFoundError: 
        error_msg = f"[Errore: File JSON dei sottotitoli non trovato (eccezione imprevista) a {str(subtitles_file_path) if 'subtitles_file_path' in locals() else 'PATH_NON_DEFINITO'}]"
        logger.error(error_msg + f" (per episode_code: {episode_code_for_log})")
        return error_msg
    except Exception as e:
        error_msg = f"[Errore: Eccezione imprevista '{type(e).__name__}' durante il caricamento dei sottotitoli - {e}]"
        logger.error(error_msg + f" (per episode_code: {episode_code_for_log})", exc_info=True)
        return error_msg

def call_llm(prompt_text: str, service: AIModelsService, llm_type: LLMType, episode_code_for_logging: str = "N/A") -> Dict[str, Any]:
    logger.info(f"Chiamata LLM per episode_code: {episode_code_for_logging} con tipo LLM: {llm_type.name}")
    # logger.debug(f"Prompt inviato all'LLM:\n{prompt_text}") # Consider uncommenting for deep debugging

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
            raw_response_str = service.call_llm(prompt_text, llm_type, with_json_extraction=True)
            logger.debug(f"Risposta grezza dall'LLM (tentativo {attempt + 1}) per {episode_code_for_logging}: {raw_response_str}")

            try:
                if isinstance(raw_response_str, dict): # Already a dict, AIModelsService might do this
                    response_data = raw_response_str
                elif isinstance(raw_response_str, str):
                    # Remove markdown ```json ... ``` if present
                    cleaned_response_str = re.sub(r"^```json\s*|\s*```$", "", raw_response_str.strip(), flags=re.MULTILINE)
                    if not cleaned_response_str:
                        logger.warning(f"LLM ha restituito una stringa vuota dopo la pulizia per {episode_code_for_logging}. Tentativo {attempt + 1}/{max_retries}.")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay_seconds)
                            continue
                        else:
                            default_error_response["explanation"] = "LLM ha restituito una stringa vuota dopo la pulizia e tutti i tentativi."
                            return default_error_response
                    response_data = json.loads(cleaned_response_str)
                else:
                    raise TypeError(f"La risposta LLM è di tipo inatteso: {type(raw_response_str)}. Contenuto: {raw_response_str}")

                # Validate essential keys
                if not all(key in response_data for key in ["score", "explanation"]):
                    logger.warning(f"LLM ha restituito una struttura JSON incompleta per {episode_code_for_logging}: {response_data}. Tentativo {attempt + 1}/{max_retries}.")
                    if attempt == max_retries - 1:
                        default_error_response["explanation"] = f"LLM ha restituito una struttura JSON incompleta dopo i tentativi: {response_data}"
                        return default_error_response
                    else:
                        time.sleep(retry_delay_seconds)
                        continue
                
                response_data.setdefault("why_not_lower", "") # Ensure these keys exist
                response_data.setdefault("why_not_higher", "")
                logger.info(f"Risposta LLM valida ricevuta per {episode_code_for_logging} (tentativo {attempt + 1}).")
                return response_data

            except json.JSONDecodeError as json_e:
                cleaned_str_for_log = cleaned_response_str if 'cleaned_response_str' in locals() and isinstance(raw_response_str, str) else 'N/A (risposta non stringa o pulizia fallita)'
                logger.warning(f"Impossibile fare il parsing della risposta JSON dall'LLM per {episode_code_for_logging}: {json_e}. Tentativo {attempt + 1}/{max_retries}. Risposta pulita: '{cleaned_str_for_log}', Grezza: '{raw_response_str}'")
                if attempt == max_retries - 1:
                    default_error_response["explanation"] = f"Impossibile fare il parsing della risposta JSON dall'LLM dopo i tentativi: {json_e}. Risposta pulita: '{cleaned_str_for_log}', Grezza: '{raw_response_str}'"
                    return default_error_response
                else:
                    time.sleep(retry_delay_seconds)
            except TypeError as type_e: # Catch unexpected response type
                logger.warning(f"Errore di tipo durante l'elaborazione della risposta LLM per {episode_code_for_logging}: {type_e}. Tentativo {attempt + 1}/{max_retries}.")
                if attempt == max_retries - 1:
                    default_error_response["explanation"] = f"Errore di tipo durante l'elaborazione della risposta LLM dopo i tentativi: {type_e}."
                    return default_error_response
                else:
                    time.sleep(retry_delay_seconds)

        except Exception as e: # Catch errors from service.call_llm() itself
            logger.error(f"Errore durante la chiamata LLM per {episode_code_for_logging} (tentativo {attempt + 1}/{max_retries}): {e}", exc_info=True)
            if attempt < max_retries - 1:
                logger.info(f"Nuovo tentativo tra {retry_delay_seconds} secondi...")
                time.sleep(retry_delay_seconds)
            else:
                logger.error(f"Chiamata LLM fallita dopo {max_retries} tentativi per {episode_code_for_logging}.")
                default_error_response["explanation"] = f"Chiamata LLM fallita dopo i tentativi: {e}"
                return default_error_response

    logger.error(f"La funzione call_llm è terminata inaspettatamente per {episode_code_for_logging} dopo tutti i tentativi.")
    return default_error_response

def main():
    # --- INIZIO MODIFICA: Caricamento del prompt da file ---

    
    # Assicuriamoci che il percorso sia assoluto per evitare ambiguità
    if not PROMPT_FILE_PATH.is_absolute():
        PROMPT_FILE_PATH = Path(os.getcwd()) / PROMPT_FILE_PATH
        logger.info(f"Il percorso relativo del file prompt '{PROMPT_FILE_PATH.name}' è stato risolto in: {PROMPT_FILE_PATH}")

    try:
        with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as f:
            USER_PROMPT_TEMPLATE = f.read() # Questa variabile sarà usata più avanti
        logger.info(f"Prompt caricato con successo da: {PROMPT_FILE_PATH}")
    except FileNotFoundError:
        logger.error(f"ERRORE CRITICO: File del prompt '{PROMPT_FILE_PATH}' non trovato. Lo script non può continuare.")
        return # Termina l'esecuzione se il prompt non può essere caricato
    except Exception as e:
        logger.error(f"ERRORE CRITICO: Errore durante la lettura del file del prompt '{PROMPT_FILE_PATH}': {e}. Lo script non può continuare.", exc_info=True)
        return # Termina l'esecuzione in caso di altri errori di lettura
    # --- FINE MODIFICA: Caricamento del prompt da file ---

    if not INPUT_CSV_PATH:
        logger.error("INPUT_CSV_PATH non è impostato. Modifica lo script per specificare il percorso del file CSV di input.")
        return

    input_csv_file = Path(INPUT_CSV_PATH)
    if not input_csv_file.is_absolute():
        input_csv_file = Path(os.getcwd()) / input_csv_file
        logger.info(f"Il percorso relativo INPUT_CSV_PATH è stato risolto in: {input_csv_file}")

    if not input_csv_file.exists():
        logger.error(f"Il file di input CSV non è stato trovato a: {input_csv_file}")
        return

    input_dir = input_csv_file.parent
    output_csv_filename = OUTPUT_FILENAME_TEMPLATE
    output_csv_path = input_dir / output_csv_filename

    try:
        service = AIModelsService()
        logger.info(f"AIModelsService inizializzato. Modello da usare: {LLM_TYPE_TO_USE.name}")
    except Exception as e:
        logger.error(f"Impossibile inizializzare AIModelsService: {e}", exc_info=True)
        return



    output_fieldnames = ['series', 'season', 'episode', 'episode_code', 
                         'score', 'explanation', 'why_not_lower', 'why_not_higher', 
                         'subtitles_load_status', 'original_input_explanation']
    
    # --- MODIFICA: Logic for continuous update ---
    processed_episode_codes: Set[str] = set()
    output_file_exists_and_has_content = output_csv_path.exists() and output_csv_path.stat().st_size > 0

    if output_file_exists_and_has_content:
        try:
            with open(output_csv_path, mode='r', encoding='utf-8', newline='') as outfile_read:
                reader = csv.DictReader(outfile_read)
                if 'episode_code' not in reader.fieldnames:
                    logger.warning(f"Il file di output esistente {output_csv_path} non contiene la colonna 'episode_code'. Le riesecuzioni potrebbero riprocessare gli episodi.")
                else:
                    for row in reader:
                        if row.get('episode_code'):
                            processed_episode_codes.add(row['episode_code'])
            logger.info(f"Letti {len(processed_episode_codes)} episode_code già processati da {output_csv_path}")
        except Exception as e:
            logger.error(f"Errore durante la lettura dei codici episodio processati da '{output_csv_path}': {e}. Si procederà come se il file fosse vuoto.", exc_info=True)
            processed_episode_codes.clear() # Reset in case of error
            output_file_exists_and_has_content = False # Treat as if we need to write headers
    # --- FINE MODIFICA ---

    rows_to_process = []
    all_input_rows_count = 0

    try:
        with open(input_csv_file, mode='r', encoding='utf-8', newline='') as infile:
            reader = csv.DictReader(infile)
            required_input_cols = ['series', 'season', 'episode', 'episode_code', 'bool_response', 'explanation']
            missing_cols = [col for col in required_input_cols if col not in reader.fieldnames]
            if missing_cols:
                logger.error(f"Colonne mancanti nel file CSV di input ({input_csv_file}): {', '.join(missing_cols)}")
                logger.error(f"Assicurati che il CSV contenga almeno: {', '.join(required_input_cols)}")
                logger.error(f"Colonne effettivamente trovate: {reader.fieldnames}")
                return
            
            for row in reader:
                all_input_rows_count += 1
                episode_code_csv = row.get('episode_code', '').strip()

                if not episode_code_csv:
                    logger.warning(f"Riga saltata per input row {all_input_rows_count}: 'episode_code' mancante o vuoto.")
                    continue

                # --- MODIFICA: Skip if already processed ---
                if episode_code_csv in processed_episode_codes:
                    logger.info(f"Episodio '{episode_code_csv}' già processato e trovato in {output_csv_path}. Saltato.")
                    continue
                # --- FINE MODIFICA ---

                if row.get('bool_response', '').strip().lower() == 'yes':
                    if not all(row.get(key) for key in ['series', 'season', 'episode']): # explanation can be empty
                        logger.warning(f"Riga saltata per episode_code '{episode_code_csv}' a causa di valori mancanti per 'series', 'season', or 'episode' nel CSV.")
                        continue
                    rows_to_process.append(row)
    except FileNotFoundError:
        logger.error(f"File di input non trovato: {input_csv_file}")
        return
    except Exception as e:
        logger.error(f"Errore durante la lettura del file CSV di input '{input_csv_file}': {e}", exc_info=True)
        return

    if not rows_to_process:
        logger.info(f"Nessuna nuova riga da processare (bool_response='Yes', series/season/episode validi, e non già processata) trovata in: {input_csv_file}")
        if not output_file_exists_and_has_content: # Create empty file with headers if it doesn't exist
            try:
                with open(output_csv_path, mode='w', encoding='utf-8', newline='') as outfile:
                    writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
                    writer.writeheader()
                logger.info(f"File di output vuoto creato con le intestazioni: {output_csv_path}")
            except IOError as e:
                logger.error(f"Errore durante la scrittura del file CSV di output vuoto '{output_csv_path}': {e}", exc_info=True)
        return
        
    logger.info(f"Trovate {len(rows_to_process)} NUOVE righe da elaborare dal file '{input_csv_file}'. L'output sarà salvato/aggiunto a '{output_csv_path}'.")

    try:
        # --- MODIFICA: Open in append mode if file exists and has content ---
        with open(output_csv_path, mode='a' if output_file_exists_and_has_content else 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
            if not output_file_exists_and_has_content:
                writer.writeheader()
            # --- FINE MODIFICA ---

            for i, row_data in enumerate(rows_to_process):
                episode_code = row_data['episode_code'] # Already checked for presence
                series_name_csv = row_data['series'] 
                season_num_csv = row_data['season']   
                episode_num_csv = row_data['episode'] 
                # --- MODIFICA: Get original explanation from input CSV ---
                original_input_explanation = row_data.get('explanation', '') # Default to empty string if missing
                # --- FINE MODIFICA ---
                
                logger.info(f"Elaborazione nuova riga {i+1}/{len(rows_to_process)}: Episode Code CSV='{episode_code}', Serie='{series_name_csv}', S={season_num_csv}, E={episode_num_csv}")
                
                subtitles_text = load_subtitles_text(
                    series_name=series_name_csv,
                    season_num_str=season_num_csv,
                    episode_num_str=episode_num_csv,
                    base_path_str=SUBTITLES_JSON_BASE_PATH,
                    episode_code_for_log=episode_code
                )
                subtitles_load_status = "OK"

                if subtitles_text.startswith("[Errore:"):
                    logger.warning(f"Errore durante il caricamento dei sottotitoli per '{episode_code}': {subtitles_text}")
                    subtitles_load_status = subtitles_text
                    
                    error_output_row = {
                        'series': series_name_csv, 'season': season_num_csv, 'episode': episode_num_csv, 
                        'episode_code': episode_code, 'score': 'ERROR_SUBTITLES', 
                        'explanation': subtitles_text, 'why_not_lower': '', 'why_not_higher': '',
                        'subtitles_load_status': subtitles_load_status, 
                        'original_input_explanation': original_input_explanation
                    }
                    writer.writerow(error_output_row)
                    logger.info(f"Riga {i+1} ({episode_code}) saltata a causa di errore caricamento sottotitoli. Scritto errore su CSV.")
                    continue
                
                try:
                    # --- MODIFICA: Pass original_input_explanation to prompt ---
                    prompt_text = USER_PROMPT_TEMPLATE.format(
                        subtitles_text=subtitles_text,
                        previous_analysis_context=original_input_explanation # Pass it here
                    )
                    # --- FINE MODIFICA ---
                except KeyError as e:
                    logger.error(f"Errore critico nella formattazione del prompt per {episode_code}: manca la chiave {e}. Questo non dovrebbe accadere.")
                    error_output_row = {
                        'series': series_name_csv, 'season': season_num_csv, 'episode': episode_num_csv,
                        'episode_code': episode_code, 'score': 'ERROR_PROMPT_FORMAT',
                        'explanation': f'Errore critico nella formattazione del prompt: manca la chiave {e}',
                        'why_not_lower': '', 'why_not_higher': '',
                        'subtitles_load_status': 'PROMPT_FORMAT_ERROR', 
                        'original_input_explanation': original_input_explanation
                    }
                    writer.writerow(error_output_row)
                    logger.info(f"Riga {i+1} ({episode_code}) saltata a causa di errore formattazione prompt. Scritto errore su CSV.")
                    continue

                llm_response = call_llm(prompt_text, service, LLM_TYPE_TO_USE, episode_code_for_logging=episode_code)

                output_row = {
                    'series': series_name_csv, 'season': season_num_csv, 'episode': episode_num_csv,
                    'episode_code': episode_code,
                    'score': llm_response.get('score', 'LLM_RESPONSE_MISSING_SCORE'),
                    'explanation': llm_response.get('explanation', 'LLM_RESPONSE_MISSING_EXPLANATION'),
                    'why_not_lower': llm_response.get('why_not_lower', ''),
                    'why_not_higher': llm_response.get('why_not_higher', ''),
                    'subtitles_load_status': subtitles_load_status,
                    'original_input_explanation': original_input_explanation
                }
                writer.writerow(output_row)
                logger.info(f"Riga {i+1} ({episode_code}) elaborata e scritta su CSV. Score: {output_row['score']}")
        
        logger.info(f"Elaborazione completata. I risultati sono stati salvati/aggiunti in: {output_csv_path}")

    except IOError as e:
        logger.error(f"Errore durante la scrittura del file CSV di output '{output_csv_path}': {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Si è verificato un errore imprevisto durante l'elaborazione principale: {e}", exc_info=True)

if __name__ == "__main__":
    main()