from src.subtitles_managing.subtitles_pipeline import process_subtitles_folder_to_jsons
from src.config.settings import SUBTITLES_DIRECTORY, LLM_OUTPUT_DIRECTORY, SRT_TO_DIALOGUE_JSON_DIRECTORY
from src.ai_interface.ai_functions_analyze_episodes import analyze_episode_dialogues_with_average
from src.ai_interface.AiModels import LLMType 
from src.reporting.reporting import print_results_table
from src.analysis.average import calculate_episode_average_score
from src.analysis.data_export import export_results_to_csv, get_average_scores_by_model, get_average_scores_by_episode
from pathlib import Path
import pandas as pd



def main():
    # Inizializza la directory di output
    llm_output_dir = LLM_OUTPUT_DIRECTORY
    llm_output_dir.mkdir(exist_ok=True)
    
    # Chiedi all'utente se vuole passare direttamente all'analisi di reliability
    print("\n=== Menu Iniziale ===\n")
    print("1. Esegui il flusso completo (elaborazione sottotitoli, analisi LLM e reliability)")
    print("2. Passa direttamente all'analisi di reliability")
    
    choice = input("\nSeleziona un'opzione (1-2): ")
    
    if choice == "1":
        # Esegui il flusso completo
        # Directory contenente i file.srt
        srt_directory = SUBTITLES_DIRECTORY
        
        # Directory di output per i file JSON
        json_subtitles_directory = SRT_TO_DIALOGUE_JSON_DIRECTORY

        process_subtitles_folder_to_jsons(srt_directory, json_subtitles_directory)

        for llm_type in LLMType:
            analyze_episode_dialogues_with_average(llm_type, json_subtitles_directory, llm_output_dir)
    
    # Chiedi all'utente quali LLM includere nel calcolo della media aggregata per episodio
    print("\nSeleziona quali LLM includere nel calcolo della media aggregata per episodio:")
    print("Opzioni disponibili:")
    
    # Mostra le opzioni disponibili
    llm_options = {i+1: llm_type for i, llm_type in enumerate(LLMType)}
    for num, llm_type in llm_options.items():
        print(f"{num}. {llm_type.value}")
    print(f"{len(llm_options)+1}. Tutti i modelli")
    
    # Ottieni la selezione dell'utente
    selected_llms = []
    try:
        selection = input("\nInserisci i numeri dei modelli da includere (separati da virgola) o 'tutti': ")
        
        if selection.lower() in ['tutti', 'all', str(len(llm_options)+1)]:
            # Usa tutti i modelli
            selected_llms = None
        else:
            # Usa solo i modelli selezionati
            selected_indices = [int(idx.strip()) for idx in selection.split(',')]
            selected_llms = [llm_options[idx] for idx in selected_indices if idx in llm_options]
    except (ValueError, KeyError):
        print("Selezione non valida, verranno utilizzati tutti i modelli.")
        selected_llms = None
    
    # Calcola la media aggregata per episodio
    calculate_episode_average_score(selected_llms)
    
    # Stampa la tabella riassuntiva solo con i modelli selezionati
    print_results_table(selected_llms)
    
    # Chiedi all'utente se vuole esportare i risultati in CSV
    export_choice = input("\nVuoi esportare i risultati in un file CSV? (s/n): ")
    if export_choice.lower() in ['s', 'si', 'sì', 'y', 'yes']:
        # Esporta i risultati in CSV
        csv_path = export_results_to_csv(llm_output_dir)
        
        # Crea un dataframe con i risultati
        df = pd.read_csv(csv_path)
        
        # Calcola e mostra i punteggi medi per modello
        avg_by_model = get_average_scores_by_model(df)
        print("\nPunteggi medi per modello e episodio:")
        print(avg_by_model)
        
        # Calcola e mostra i punteggi medi per episodio
        avg_by_episode = get_average_scores_by_episode(df)
        print("\nPunteggi medi per episodio:")
        print(avg_by_episode)

if __name__ == "__main__":
    main()