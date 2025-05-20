# subtitles_pipeline.py

import json
import logging
import re
from pathlib import Path
from .SubtitlesManagingModels import DialogueLine

# Setup basic logging (adjust level and format as needed)
# Adding filename and line number for better debugging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(filename)s:%(lineno)d: %(message)s')

def _clean_srt_text(text: str) -> str:
    """
    Cleans text extracted from an SRT block.
    - Joins multiple lines with spaces.
    - Removes HTML-like tags (e.g., <font>, <i>).
    - Collapses multiple whitespace characters into single spaces.
    - Strips leading/trailing whitespace.
    """
    # 1. Join lines within the block with a space (handle potential \n)
    #    Splitlines handles different line endings (\n, \r, \r\n)
    lines = [line.strip() for line in text.splitlines()]
    cleaned = ' '.join(line for line in lines if line) # Join non-empty lines

    # 2. Remove common SRT tags (like <font>, <i>, <b>, <u>, etc.)
    #    This regex removes anything enclosed in < >
    cleaned = re.sub(r'<[^>]+>', '', cleaned)

    # 3. Collapse multiple spaces/whitespace characters into one
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # 4. Optional: Decode HTML entities (like & -> &, < -> <) if needed
    # import html
    # cleaned = html.unescape(cleaned)

    return cleaned

def _parse_srt_file(filepath: Path) -> list[DialogueLine]:
    """
    Parses an SRT file line by line, handling multiple encodings and cleaning text.
    Returns a list of DialogueLine objects or an empty list on failure.
    """
    encodings_to_try = [
        'utf-8-sig',    # 1. UTF-8 with BOM (Notepad often adds this)
        'utf-8',        # 2. UTF-8 without BOM (Very common)
        # --- Then try UTF-16 with BOM ---
        'utf-16',       # 3. UTF-16 with BOM (LE or BE)
        # --- Then legacy/less common formats ---
        'cp1252',       # 4. Common Western Windows legacy
        'latin-1',      # 5. ISO-8859-1 (Western Europe)
        # --- Then UTF-16 without BOM (less common than UTF-8) ---
        'utf-16-le',    # 6. UTF-16 Little Endian (no BOM)
        'utf-16-be',    # 7. UTF-16 Big Endian (no BOM)
        # --- Then Common CJK Encodings ---
        'gbk',          # 8. Simplified Chinese
        'big5',         # 9. Traditional Chinese
        'shift-jis',    # 10. Japanese
        'euc-jp',       # 11. Japanese (Unix)
        'cp932',        # 12. Japanese (Windows)
        'euc-kr',       # 13. Korean
        'cp949',        # 14. Korean (Windows)
        # Add others if needed
    ]
    lines = None
    detected_encoding = None

    # --- 1. Read File with Encoding Detection ---
    for enc in encodings_to_try:
        try:
            logging.debug(f"Attempting to read {filepath.name} with encoding: {enc}") # Add this line
            with open(filepath, 'r', encoding=enc) as file:
                # Read the whole content at once to ensure full decode attempt
                content = file.read()
            # If read succeeds, split into lines
            lines = content.splitlines()
            detected_encoding = enc
            logging.info(f"Successfully read {filepath.name} with encoding: {enc}") # Changed from debug to info
            break # Exit loop on successful read

        # --- Refined Exception Handling ---
        except UnicodeError as ue: # Catch broader Unicode errors including Decode/Encode
            # This specifically targets errors during the decoding process.
            # The "BOM not found" error *should* fall under this.
            logging.debug(f"Decoding failed for {filepath.name} with {enc}: {ue}. Trying next...")
            # Make sure lines is reset if decoding fails partially (though read() usually fails fully)
            lines = None
            detected_encoding = None
            continue # Definitely try the next encoding

        except FileNotFoundError:
            logging.error(f"File not found: {filepath}")
            return [] # Stop processing this specific file if it doesn't exist

        except OSError as ose:
            # Catch other OS level errors like permission denied
            logging.error(f"OS error accessing {filepath.name}: {ose}")
            return [] # Stop processing this file for OS errors

        except Exception as e:
            # Catch any OTHER unexpected error during open/read.
            # This might indicate a deeper problem. Log it as an error.
            logging.error(f"Unexpected error during file read operation for {filepath.name} with {enc}: {e}", exc_info=True)
            # Reset state just in case
            lines = None
            detected_encoding = None
            # We will still continue to try other encodings, just in case
            # it was some edge case related to the specific encoding attempt.
            continue

    # --- Check if any encoding succeeded ---
    if lines is None:
        logging.error(f"Could not decode {filepath.name} after trying all encodings: {encodings_to_try}")
        return [] # Return empty list if all attempts failed

    # --- 2. Parse Content Line by Line (using the 'lines' variable) ---
    # ... (rest of the parsing logic remains the same) ...

    entries = []
    buffer = []
    block_start_line_num = 1 # For logging errors
    current_line_num = 0 # Initialize for potential error logging below
    try:
        # (parsing loop as before)
        for current_line_num, line in enumerate(lines, 1):
             if line.strip() == "": # End of a block
                 if buffer:
                     # --- Process the completed block in the buffer ---
                     try:
                         # Expecting: Number, Timecode, Text line(s)
                         if len(buffer) >= 3 and buffer[0].strip().isdigit() and '-->' in buffer[1]:
                             line_number_str = buffer[0].strip()
                             timecode = buffer[1].strip()
                             raw_text = "\n".join(buffer[2:])
                             cleaned_text = _clean_srt_text(raw_text)
                             start_str, end_str = [tc.strip() for tc in timecode.split('-->')]
                             entry = DialogueLine(line_number_str, start_str, end_str, cleaned_text)
                             entries.append(entry)
                         else:
                              if any(b.strip() for b in buffer):
                                 logging.warning(f"Skipping malformed block in {filepath.name} ending near line {current_line_num}. Content: {buffer}")
                     except ValueError as ve:
                         logging.warning(f"Skipping block in {filepath.name} ending near line {current_line_num} due to data error: {ve}. Content: {buffer}")
                     except Exception as e:
                          logging.warning(f"Skipping block in {filepath.name} ending near line {current_line_num} due to unexpected error: {e}. Content: {buffer}")
                     finally:
                          buffer = []
                          block_start_line_num = current_line_num + 1
             else:
                  if not buffer:
                      block_start_line_num = current_line_num
                  buffer.append(line)

        # --- 3. Handle the very last block ---
        if buffer:
            try:
                  # (logic as before)
                  if len(buffer) >= 3 and buffer[0].strip().isdigit() and '-->' in buffer[1]:
                       line_number_str = buffer[0].strip()
                       timecode = buffer[1].strip()
                       raw_text = "\n".join(buffer[2:])
                       cleaned_text = _clean_srt_text(raw_text)
                       start_str, end_str = [tc.strip() for tc in timecode.split('-->')]
                       entry = DialogueLine(line_number_str, start_str, end_str, cleaned_text)
                       entries.append(entry)
                  else:
                       if any(b.strip() for b in buffer):
                          logging.warning(f"Skipping malformed final block in {filepath.name} starting near line {block_start_line_num}. Content: {buffer}")
            except ValueError as ve:
                  logging.warning(f"Skipping final block in {filepath.name} starting near line {block_start_line_num} due to data error: {ve}. Content: {buffer}")
            except Exception as e:
                  logging.warning(f"Skipping final block in {filepath.name} starting near line {block_start_line_num} due to unexpected error: {e}. Content: {buffer}")

    except Exception as e:
        logging.error(f"Major error processing content of {filepath.name} (Decoded with: {detected_encoding}) near line {current_line_num}: {e}", exc_info=True)
        return []

    if not entries:
         logging.warning(f"Parsing completed for {filepath.name} (Decoded with: {detected_encoding}), but no valid subtitle entries were found.")

    return entries


def process_subtitles_folder_to_jsons(srt_directory: Path, output_directory: Path):
    """
    Processes all .srt files in a directory, parsing them and saving as JSON files.
    """
    if not isinstance(srt_directory, Path):
         srt_directory = Path(srt_directory)
    if not isinstance(output_directory, Path):
         output_directory = Path(output_directory)

    output_directory.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    srt_files = list(srt_directory.glob("*.srt"))
    logging.info(f"Found {len(srt_files)} SRT files in '{srt_directory}'.")
    if not srt_files:
        logging.warning("No .srt files found to process.")
        return

    processed_count = 0
    failed_count = 0
    for srt_file in srt_files:

        # Saving to JSON
        output_file = output_directory / (srt_file.stem + ".json")

        if output_file.exists():
            logging.warning(f"Skipping {srt_file.name} as JSON output already exists: {output_file}")
            continue # Skip to the next file

        logging.info(f"--- Processing {srt_file.name} ---")
        try:
            # Use the robust line-by-line parser
            dialogue_entries = _parse_srt_file(srt_file)

            if not dialogue_entries:
                # _parse_srt_file logs specific reasons if possible
                logging.warning(f"No valid entries parsed from {srt_file.name}. Skipping JSON output.")
                failed_count += 1
                continue # Skip to the next file

            # Conversion to list of dictionaries
            json_data = [entry.to_dict() for entry in dialogue_entries]

            
            logging.info(f"Attempting to save {len(json_data)} entries to: {output_file}")

            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=4)
                logging.info(f"Successfully saved: {output_file}")
                processed_count += 1
            except IOError as ioe:
                 logging.error(f"!!! FAILED to write JSON for {srt_file.name} to {output_file}: {ioe}", exc_info=True)
                 failed_count += 1
            except Exception as json_e: # Catch potential json.dump errors
                 logging.error(f"!!! FAILED during JSON serialization for {srt_file.name}: {json_e}", exc_info=True)
                 failed_count += 1


        except Exception as e:
            # Catch unexpected errors during the processing loop for THIS file
            logging.error(f"!!! UNEXPECTED error processing {srt_file.name}: {e}", exc_info=True)
            failed_count += 1

    logging.info(f"--- Processing Finished ---")
    logging.info(f"Successfully processed: {processed_count} files.")
    logging.info(f"Failed/Skipped: {failed_count} files.")