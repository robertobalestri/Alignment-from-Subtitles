#!/usr/bin/env python3
import os
import re
import shutil

# --- Configuration ---
# Adjust these paths if necessary
input_base_dir = "subtitles_raw_US" # Use forward slash even on Windows for compatibility
output_base_dir = "subtitles_dataset_US_renamed_structured" # Renamed output dir for clarity

# Mapping from folder name (Series Name) to abbreviation
series_map = {
    "ER": "ER",
}

# --- Regular Expressions ---
# Compile regex patterns for efficiency
pattern_se = re.compile(r'S(\d{1,2})E(\d{1,2})', re.IGNORECASE)
pattern_sx = re.compile(r'S(\d{1,2})[xX](\d{1,2})', re.IGNORECASE)
pattern_x = re.compile(r'(\d{1,2})[xX](\d{1,2})', re.IGNORECASE)

# --- Main Logic ---

def rename_subtitles():
    """
    Finds, parses, renames, and copies subtitle files
    into a NEW structured output directory: Output/Abbr/S##/E##/AbbrS##E##.srt.
    Also tracks and reports errors.
    """
    print(f"Starting subtitle renaming process.")
    print(f"Input directory: '{os.path.normpath(input_base_dir)}'")
    print(f"Output directory: '{os.path.normpath(output_base_dir)}'")
    print("Output will use NEW structure: Abbr/S##/E##/")

    processed_files = 0
    error_files = [] # List to store paths of files with errors

    # Walk through the input directory structure
    for dirpath, dirnames, filenames in os.walk(input_base_dir):

        # Check if the current directory contains .srt files
        srt_files = [f for f in filenames if f.lower().endswith('.srt')]
        if not srt_files:
            continue # Skip directories without .srt files

        # --- Determine Series Name and Abbreviation ---
        # Logic remains the same as the previous version
        series_abbr = None
        series_folder_name = None
        try:
            path_parts = os.path.normpath(dirpath).split(os.sep)
            input_base_parts_len = len(os.path.normpath(input_base_dir).split(os.sep))
            for i in range(input_base_parts_len, len(path_parts)):
                part = path_parts[i]
                if part in series_map:
                    series_folder_name = part
                    series_abbr = series_map[series_folder_name]
                    break

            if not series_abbr:
                 current_base_name = os.path.basename(os.path.normpath(dirpath))
                 if current_base_name in series_map:
                     series_folder_name = current_base_name
                     series_abbr = series_map[series_folder_name]

            if not series_abbr:
                print(f"Warning: Could not determine known series name for path '{os.path.normpath(dirpath)}'. Skipping files within.")
                continue

        except Exception as e:
            print(f"Warning: Error determining series name for path '{os.path.normpath(dirpath)}'. Error: {e}. Skipping directory.")
            continue
        # --- End Series Name Determination ---

        # Process each .srt file found in the current directory
        for filename in srt_files:
            original_full_path = os.path.join(dirpath, filename)
            season_num = None
            episode_num = None
            parsed_successfully = False # Flag to check if parsing worked

            # --- Try matching patterns in order ---
            match = pattern_se.search(filename)
            if match:
                try:
                    season_num = int(match.group(1))
                    episode_num = int(match.group(2))
                    parsed_successfully = True
                except ValueError:
                    print(f"Warning: Found SxxExx pattern in '{filename}' but failed number conversion.")

            if not parsed_successfully:
                 match = pattern_sx.search(filename)
                 if match:
                     try:
                         season_num = int(match.group(1))
                         episode_num = int(match.group(2))
                         parsed_successfully = True
                     except ValueError:
                         print(f"Warning: Found SxxXxx pattern in '{filename}' but failed number conversion.")

            if not parsed_successfully:
                match = pattern_x.search(filename)
                if match:
                    try:
                        season_num = int(match.group(1))
                        episode_num = int(match.group(2))
                        parsed_successfully = True
                    except ValueError:
                         print(f"Warning: Found xxXxx pattern in '{filename}' but failed number conversion.")
            # --- End Pattern Matching ---

            # --- Proceed only if parsing was fully successful ---
            if parsed_successfully and season_num is not None and episode_num is not None:
                # Format the new filename (remains the same)
                new_filename = f"{series_abbr}S{season_num:02d}E{episode_num:02d}.srt"

                try:
                    # --- Create NEW structured directory path: Abbr/S##/E## ---
                    season_folder_name = f"S{season_num:02d}"
                    episode_folder_name = f"E{episode_num:02d}"

                    # Construct the target directory path based on the new structure
                    destination_dir = os.path.join(output_base_dir, series_abbr, season_folder_name, episode_folder_name)

                    # Ensure this destination directory exists, creating intermediate ones if needed
                    os.makedirs(destination_dir, exist_ok=True)

                    # Construct the full path for the destination file within the new structure
                    destination_full_path = os.path.join(destination_dir, new_filename)
                    # --- End structure creation ---

                    # Copy the file to the new location
                    shutil.copy2(original_full_path, destination_full_path)

                    # Display relative paths for clarity (original vs NEW structure)
                    display_orig_rel_path = os.path.relpath(original_full_path, input_base_dir)
                    display_new_rel_path = os.path.relpath(destination_full_path, output_base_dir)
                    print(f"  Copied: '{os.path.normpath(display_orig_rel_path)}' -> '{os.path.normpath(display_new_rel_path)}'")
                    processed_files += 1
                except Exception as e:
                    print(f"Error copying '{os.path.normpath(original_full_path)}' to '{os.path.normpath(destination_full_path)}': {e}")
                    if original_full_path not in error_files:
                        error_files.append(original_full_path)
            else:
                # If parsing failed
                print(f"Warning: Could not extract valid season/episode info from: '{filename}' in '{os.path.normpath(dirpath)}'. Skipping.")
                if original_full_path not in error_files:
                     error_files.append(original_full_path)
            # --- End Processing Block ---

    # --- Print Summary ---
    print("\n--- Renaming Complete ---")
    print(f"Successfully processed and copied: {processed_files} files.")
    actual_skipped_count = len(error_files)
    print(f"Skipped due to errors or parsing issues: {actual_skipped_count} files.")
    print(f"Output files are located in: '{os.path.normpath(output_base_dir)}' (using Abbr/S##/E## structure)")

    # --- Print Error List ---
    if error_files:
        print("\n--- Files with Errors or Parsing Issues (Original Paths) ---")
        error_files.sort()
        for filepath in error_files:
            print(f"- {os.path.normpath(filepath)}")
    else:
        print("\nNo files were skipped due to processing errors.")

# --- Run the script ---
if __name__ == "__main__":
    # Basic check if input directory exists
    if not os.path.isdir(input_base_dir):
        print(f"Error: Input directory '{os.path.normpath(input_base_dir)}' not found.")
        print(f"       Please check the 'input_base_dir' variable in the script.")
    else:
        rename_subtitles()