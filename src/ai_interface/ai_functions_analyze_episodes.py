import time
from pathlib import Path
import json
import statistics
from .AiModels import LLMType, AIModelsService


# Base prompt remains the same for all LLM types
base_prompt = """
    I need you to evaluate the alignment of a text—a list of dialogues from an episode of a TV series—with respect to the theme of abortion. To do this, I have provided you with the episode's dialogues, which you are to analyze using a Likert-like scale regarding the polarity of judgment on abortion. I have included the scale below.
    In the episode, there are of course various positions on abortion, expressed by different characters. What I need is for you to identify the dominant position within the text and place it on the scale.

    You must return your response as **valid JSON** with the following structure:
    {
    "detailed_evaluation": "...your explanation here...",
    "score": ...a number from -3 to +3...
    }

    Here is the scale:
    \"\"\"
    Abortion Discourse Scale (-3 to +3)
    -3 (Fully Pro-Life)
    * Overarching emphasis on the fetus’s moral status as an absolute imperative, making abortion nearly always unacceptable.
    * Emotional, social, or medical considerations for the pregnant individual are overshadowed by the duty to preserve potential life.
    * The text frames abortion as ethically inadmissible, sometimes portraying medical or emotional appeals as secondary or misplaced.
    -2 (Strongly Pro-Life)
    * The text strongly advocates protecting fetal life, yet may offer limited acknowledgment of the pregnant individual’s emotional or social context.
    * Ethical arguments about the fetus’s moral worth dominate, and alternative viewpoints are discussed minimally or with strong reservation.
    -1 (Moderately Pro-Life)
    * The text emphasizes the fetus’s moral value but devotes some attention to the individual’s emotional, social, or medical considerations.
    * Exceptions (e.g., life-threatening medical cases) are recognized, although these are presented as rare or ethically burdensome.
    * The stance remains clearly pro-life but with limited space for contextual complexity.
    0 (Neutral / Balanced)
    * The text presents multiple angles—ethical, emotional, social, medical—without definitively favoring either “pro-life” or “pro-choice.”
    * It explores potential conflicts between preserving fetal life and respecting bodily autonomy, acknowledging that both perspectives have weight.
    * No strong bias is evident; the text strives for an even-handed examination.
    +1 (Moderately Pro-Choice)
    * The text generally supports the pregnant individual’s autonomy and focuses on personal, emotional, and social factors influencing the decision.
    * Fetal moral status may be acknowledged, but does not override the primacy of individual choice in most scenarios.
    * The overall orientation leans toward validating the option to end a pregnancy when justified by context.
    +2 (Strongly Pro-Choice)
    * The text portrays abortion as a legitimate exercise of reproductive freedom, highlighting the individual’s well-being and the broader social or emotional ramifications.
    * Fetal considerations are noted but typically subordinated to arguments about personal autonomy and rights.
    * Disagreement with abortion is recognized but not foregrounded.
    +3 (Fully Pro-Choice)
    * The text unequivocally affirms abortion as a fundamental right, with emotional, social, and medical aspects strongly reinforcing the individual’s autonomy.
    * Any moral status of the fetus is seen as secondary or negligible compared to the pregnant individual’s needs and choices.
    * Abortion is framed as ethically, socially, and personally justified under virtually all circumstances.
    How to Use This Scale
    * When analyzing a text (e.g., a scene in a medical drama or any narrative), consider how it frames the ethical debate (moral arguments), acknowledges emotional factors (e.g., family bonds, fear, psychological distress), reflects social dynamics (support systems, social pressures, cultural attitudes), and weighs medical considerations (risks to health, viability, etc.).
    * Choose the numeric value (-3 to +3) that best represents the text’s overall stance, taking into account all relevant dimensions (not only moral).
    * This single scale allows you to categorize texts on a continuum while still noting the multi-faceted arguments they present.
    \"\"\"

    REMEMBER TO OUTPUT A VALID JSON WITHOUT ANY OTHER TEXT OUTSIDE OF IT.

    Here are the episode's dialogues:
    """

client = AIModelsService()

def analyze_episode_dialogues_with_average(
    llm_type: LLMType,
    json_dir: Path = Path("/work/subtitles/json"),
    response_dir: Path = Path("/work/subtitles/response"),
    attempts: int = 5
):
    """
    Analyze TV series episode dialogues with multiple attempts for average calculation.
    
    :param llm_type: The type of Large Language Model to use
    :param json_dir: Directory containing subtitle JSON files
    :param response_dir: Directory to save analysis results
    :param attempts: Number of attempts for average calculation
    """
    # Ensure response directory exists
    response_dir.mkdir(exist_ok=True)

    # Process each JSON subtitle file
    for json_file in json_dir.glob("*.json"):
        episode_name = json_file.stem
        episode_dir = response_dir / episode_name
        episode_dir.mkdir(exist_ok=True)
        
        llm_dir = episode_dir / llm_type.value
        llm_dir.mkdir(exist_ok=True)
        
        scores = []
        
        for attempt in range(1, attempts + 1):
            attempt_file = llm_dir / f"attempt_{attempt}.json"
            
            # Skip if attempt file already exists
            if attempt_file.exists():
                try:
                    with open(attempt_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    scores.append(existing_data["score"])
                    print(f"Skipping attempt {attempt} for {episode_name} - file already exists")
                    continue
                except Exception as e:
                    print(f"Error reading existing attempt file {attempt_file}: {e}")
            
            try:
                # Load subtitles
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract dialogue texts
                dialogue_texts = [entry["text"] for entry in data]
                joined_dialogue = "\n".join(dialogue_texts)

                # Compose full prompt (same as original)
                full_prompt = base_prompt + joined_dialogue

                # Call LLM with specific type
                response_data = client.call_llm(prompt=full_prompt, llm_type=llm_type)
                
                # Save individual attempt
                with open(attempt_file, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, indent=4, ensure_ascii=False)
                
                scores.append(response_data["score"])
                
            except Exception as e:
                print(f"Attempt {attempt} failed for {json_file.name}: {e}")
                time.sleep(10)  # Wait before retrying this attemp
        
        if scores:
            avg_score = statistics.mean(scores)
            print(f"Average score by {llm_type} for {episode_name} ({llm_type.value}): {avg_score}")
            
            # Save average score
            avg_file = llm_dir / "average_score.txt"
            with open(avg_file, 'w', encoding='utf-8') as f:
                f.write(str(avg_score))
            


def analyze_episode_dialogues(
    llm_type: LLMType, 
    json_dir: Path = Path("/work/subtitles/json"), 
    response_dir: Path = Path("/work/subtitles/response")
):
    """
    Analyze TV series episode dialogues for abortion discourse using specified LLM.
    
    :param llm_type: The type of Large Language Model to use
    :param json_dir: Directory containing subtitle JSON files
    :param response_dir: Directory to save analysis results
    """
    # Ensure response directory exists
    response_dir.mkdir(exist_ok=True)

    # Process each JSON subtitle file
    for json_file in json_dir.glob("*.json"):
        print(f"Analyzing: {json_file.name}")

        try:
            # Load subtitles
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract dialogue texts
            dialogue_texts = [entry["text"] for entry in data]
            joined_dialogue = "\n".join(dialogue_texts)

            # Compose full prompt
            full_prompt = base_prompt + joined_dialogue

            raw_response = ""

            # Attempt to get valid JSON response
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    print(f"Attempt {attempt} for {json_file.name}...")
                    
                    # Call LLM with specific type
                    response_data = client.call_llm(prompt=full_prompt, llm_type=llm_type)

                    raw_response = response_data

                    response_data["episode"] = json_file.stem
                    response_data["llm_type"] = llm_type.value

                    # Generate filename with LLM type
                    response_filename = response_dir / f"{json_file.stem}_{llm_type.value}.json"
                    
                    with open(response_filename, 'w', encoding='utf-8') as out_file:
                        json.dump(response_data, out_file, indent=4, ensure_ascii=False)

                    print(f"Response saved to: {response_filename}\n{'-'*80}\n")
                    break  # Exit attempts loop if successful

                except Exception as e:
                    print(f"Invalid response (attempt {attempt}): ", e)
                    
                    if attempt == max_attempts:
                        # Save raw response if all attempts fail
                        fallback_filename = response_dir / f"{json_file.stem}_{llm_type.value}_RAW.txt"
                        with open(fallback_filename, 'w', encoding='utf-8') as f_raw:
                            f_raw.write(raw_response)
                        print(f"Raw response saved to: {fallback_filename}\n{'-'*80}\n")
                    
                    time.sleep(20)  # Wait between attempts

            # Short pause between files
            time.sleep(5)
        
        except Exception as e:
            print(f"Unhandled error with file {json_file.name}: {e}")