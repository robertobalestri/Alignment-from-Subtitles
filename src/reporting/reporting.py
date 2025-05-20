from pathlib import Path

def print_results_table(selected_llms=None):
    """
    Stampa una tabella riassuntiva dei risultati per ogni episodio
    con una formattazione migliorata per una migliore leggibilità
    
    :param selected_llms: Lista opzionale di LLMType da includere nella tabella. Se None, include tutti i LLM.
    """
    # Definizione delle larghezze delle colonne
    col_widths = {
        "episode": 12,
        "llm": 15,
        "model_score": 15,
        "episode_avg": 15
    }
    
    # Intestazione della tabella con stile migliorato
    print("\n" + "=" * 60)
    print("  RISULTATI ANALISI DIALOGHI  ".center(60))
    print("=" * 60)
    
    # Intestazioni delle colonne formattate
    header = f"│ {'Episodio'.ljust(col_widths['episode'])} │ {'LLM'.ljust(col_widths['llm'])} │ {'Score Modello'.ljust(col_widths['model_score'])} │ {'Media Episodio'.ljust(col_widths['episode_avg'])} │"
    separator = f"├─{'─' * col_widths['episode']}─┼─{'─' * col_widths['llm']}─┼─{'─' * col_widths['model_score']}─┼─{'─' * col_widths['episode_avg']}─┤"
    
    print(f"┌─{'─' * col_widths['episode']}─┬─{'─' * col_widths['llm']}─┬─{'─' * col_widths['model_score']}─┬─{'─' * col_widths['episode_avg']}─┐")
    print(header)
    print(separator)
    
    # Percorso della directory di output
    output_dir = Path("output")
    
    # Sort episodes naturally
    episodes = sorted(output_dir.glob("*"), key=lambda x: x.name)
    
    # Sort LLMs using natural sorting
    for episode_dir in episodes:
        if not episode_dir.is_dir():
            continue
            
        episode_name = episode_dir.name
        
        # Leggi il punteggio medio aggregato dell'episodio (average_score.txt nella cartella episodio)
        episode_score_file = episode_dir / "average_score.txt"
        episode_avg_score = "N/A"
        if episode_score_file.exists():
            with open(episode_score_file, 'r') as f:
                score_text = f.read().strip()
                # Add validation for empty files
                if not score_text:
                    episode_avg_score = "ERR: Empty"
                    continue
                try:
                    score = float(score_text)
                    # Updated validation for -3 to +3 range
                    if not -3 <= score <= 3:
                        episode_avg_score = "ERR: Range"
                    else:
                        episode_avg_score = f"{score:.2f}"
                except ValueError:
                    episode_avg_score = "ERR: Format"
        
        # Per ogni LLM
        for llm_dir in episode_dir.glob("*"):
            if not llm_dir.is_dir():
                continue
                
            llm_name = llm_dir.name
            
            # Se è stata specificata una lista di LLM da includere, verifica che questo LLM sia nella lista
            if selected_llms is not None and llm_name not in [llm_type.value for llm_type in selected_llms]:
                continue
            
            # Leggi il punteggio medio del modello (average_score.txt nella cartella LLM)
            model_score_file = llm_dir / "average_score.txt"
            model_score = "N/A"
            if model_score_file.exists():
                with open(model_score_file, 'r') as f:
                    score_text = f.read().strip()
                    try:
                        # Formatta il punteggio con 2 decimali se è un numero
                        model_score = f"{float(score_text):.2f}"
                    except ValueError:
                        model_score = score_text
            
            # Stampa la riga formattata
            row = f"│ {episode_name.ljust(col_widths['episode'])} │ {llm_name.ljust(col_widths['llm'])} │ {model_score.ljust(col_widths['model_score'])} │ {episode_avg_score.ljust(col_widths['episode_avg'])} │"
            print(row)
    
    # Chiusura della tabella
    print(f"└─{'─' * col_widths['episode']}─┴─{'─' * col_widths['llm']}─┴─{'─' * col_widths['model_score']}─┴─{'─' * col_widths['episode_avg']}─┘")