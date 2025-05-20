import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

def read_json_file(file_path: Path) -> Dict[str, Any]:
    """Legge un file JSON e restituisce il suo contenuto."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_episode_model_attempt(file_path: Path) -> tuple:
    """Estrae l'episodio, il modello e il numero del tentativo dal percorso del file."""
    # Il percorso è del tipo: output/EPISODIO/MODELLO/attempt_N.json
    parts = file_path.parts
    episode = parts[-3]  # Terzo elemento da destra
    model = parts[-2]    # Secondo elemento da destra
    attempt = int(parts[-1].split('_')[1].split('.')[0])  # Estrae il numero dal nome del file
    return episode, model, attempt

def collect_json_results(output_dir: Path) -> pd.DataFrame:
    """Raccoglie tutti i risultati JSON in un dataframe pandas."""
    results = []
    
    # Itera su tutte le cartelle degli episodi
    for episode_dir in output_dir.iterdir():
        if not episode_dir.is_dir() or episode_dir.name.startswith('.'): 
            continue
            
        # Itera su tutte le cartelle dei modelli per questo episodio
        for model_dir in episode_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith('.'):
                continue
                
            # Itera su tutti i file JSON di tentativo per questo modello
            for attempt_file in model_dir.iterdir():
                if not attempt_file.name.startswith('attempt_') or not attempt_file.name.endswith('.json'):
                    continue
                    
                try:
                    # Leggi il file JSON
                    data = read_json_file(attempt_file)
                    
                    # Estrai le informazioni dal percorso del file
                    episode, model, attempt = extract_episode_model_attempt(attempt_file)
                    
                    # Aggiungi i dati al risultato
                    results.append({
                        'episode': episode,
                        'model': model,
                        'attempt': attempt,
                        'score': data.get('score'),
                        'detailed_evaluation': str(data.get('detailed_evaluation')).replace('\n', ' '),
                    })
                except Exception as e:
                    print(f"Errore durante l'elaborazione del file {attempt_file}: {e}")
    
    # Crea un dataframe pandas dai risultati
    df = pd.DataFrame(results)
    
    # Ordina il dataframe per episodio, modello e tentativo
    df = df.sort_values(by=['episode', 'model', 'attempt'])
    
    return df

def export_results_to_csv(output_dir: Path, csv_path: Path = None) -> Path:
    """Esporta i risultati in un file CSV.
    
    Args:
        output_dir: Directory contenente i risultati JSON
        csv_path: Percorso del file CSV di output (opzionale)
        
    Returns:
        Path: Percorso del file CSV creato
    """
    # Se non è specificato un percorso per il CSV, usa il percorso predefinito
    if csv_path is None:
        csv_path = output_dir.parent / 'results_export.csv'
    
    # Raccogli i risultati in un dataframe
    df = collect_json_results(output_dir)
    
    # Esporta il dataframe in un file CSV
    df.to_csv(csv_path, index=False)
    
    print(f"Risultati esportati in {csv_path}")
    return csv_path

def get_average_scores_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola i punteggi medi per ogni modello e episodio."""
    # Raggruppa per episodio e modello, quindi calcola la media dei punteggi
    avg_scores = df.groupby(['episode', 'model'])['score'].mean().reset_index()
    avg_scores = avg_scores.rename(columns={'score': 'average_score'})
    
    return avg_scores

def get_average_scores_by_episode(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola i punteggi medi per ogni episodio."""
    # Raggruppa per episodio e calcola la media dei punteggi
    avg_scores = df.groupby(['episode'])['score'].mean().reset_index()
    avg_scores = avg_scores.rename(columns={'score': 'average_score'})
    
    return avg_scores