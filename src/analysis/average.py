from pathlib import Path
import statistics

def calculate_episode_average_score(llm_types_to_include=None):
    """
    Calcola la media dei punteggi dei modelli per ogni episodio.

    :param llm_types_to_include: Lista di LLMType da includere nel calcolo. Se None, tutti i LLM saranno considerati.
    """
    # Percorso della directory di output
    output_dir = Path("output")
    
    # Per ogni episodio
    for episode_dir in output_dir.glob("*"):
        if not episode_dir.is_dir():
            continue
            
        episode_name = episode_dir.name
        model_scores = []
        
        # Raccogli i punteggi dei modelli per questo episodio, filtrando per i LLM selezionati
        for llm_dir in episode_dir.glob("*"):
            if not llm_dir.is_dir():
                continue
                
            # Se è stata specificata una lista di LLM da includere, verifica che questo LLM sia nella lista
            llm_name = llm_dir.name
            if llm_types_to_include is not None and llm_name not in [llm_type.value for llm_type in llm_types_to_include]:
                continue
                
            # Leggi il punteggio medio del modello (average_score.txt nella cartella LLM)
            model_score_file = llm_dir / "average_score.txt"
            if model_score_file.exists():
                try:
                    with open(model_score_file, 'r') as f:
                        score = float(f.read().strip())
                    model_scores.append(score)
                except (ValueError, FileNotFoundError) as e:
                    print(f"Errore nella lettura del file {model_score_file}: {e}")
        
        # Calcola la media dei punteggi dei modelli (punteggio medio aggregato dell'episodio)
        if model_scores:
            episode_average_score = statistics.mean(model_scores)
            print(f"Punteggio medio aggregato per {episode_name}: {episode_average_score}")
            
            # Salva il punteggio medio aggregato dell'episodio
            episode_score_file = episode_dir / "average_score.txt"
            with open(episode_score_file, 'w') as f:
                f.write(str(episode_average_score))
        else:
            print(f"Nessun punteggio di modello trovato per {episode_name}")