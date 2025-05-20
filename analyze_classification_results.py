import os
import csv
from collections import Counter

def analyze_csv_files(root_folder):
    results = {}
    target_column_name = 'bool_response' # Changed to bool_response as per requirement

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv') and 'classification_results' in file:
                file_path = os.path.join(subdir, file)
                yes_count = 0
                no_count = 0
                column_found = False
                try:
                    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                        reader = csv.DictReader(csvfile)
                        if target_column_name not in reader.fieldnames:
                            print(f"Warning: Column '{target_column_name}' not found in {file_path}. Skipping this file.")
                            results[file_path] = {'Yes': 'N/A', 'No': 'N/A', 'Error': f"Column '{target_column_name}' not found"}
                            continue
                        column_found = True
                        for row in reader:
                            value = row.get(target_column_name, '').strip().lower()
                            if value == 'yes':
                                yes_count += 1
                            elif value == 'no':
                                no_count += 1
                    if column_found:
                        results[file_path] = {'Yes': yes_count, 'No': no_count}
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    results[file_path] = {'Yes': 'Error', 'No': 'Error', 'Error': str(e)}

    if not results:
        print(f"No relevant CSV files found in {root_folder} or its subdirectories.")
        return

    print("\n--- Classification Results Summary ---")
    # Determine column widths for formatting
    max_file_path_len = max(len(os.path.relpath(fp, root_folder)) for fp in results.keys()) if results else 30
    max_file_path_len = max(max_file_path_len, len("File Path"))
    yes_col_width = max(max(len(str(r['Yes'])) for r in results.values() if 'Yes' in r), len("Yes")) + 2
    no_col_width = max(max(len(str(r['No'])) for r in results.values() if 'No' in r), len("No")) + 2

    header = f"| {'File Path':<{max_file_path_len}} | {'Yes':>{yes_col_width}} | {'No':>{no_col_width}} |"
    print(header)
    print(f"|{'-' * (max_file_path_len + 2)}|{'-' * (yes_col_width + 2)}|{'-' * (no_col_width + 2)}|")

    for file_path, counts in results.items():
        relative_path = os.path.relpath(file_path, root_folder)
        yes_val = counts.get('Yes', 'N/A')
        no_val = counts.get('No', 'N/A')
        print(f"| {relative_path:<{max_file_path_len}} | {str(yes_val):>{yes_col_width}} | {str(no_val):>{no_col_width}} |")
    print("-------------------------------------")

if __name__ == '__main__':
    # Assuming the script is located in the workspace root directory.
    workspace_root = os.path.dirname(os.path.abspath(__file__))

    target_folders_to_analyze = [
        "output_experiment_gpt4_1_IT",
        "output_experiment_gpt4_1_US"
    ]

    print(f"Script is running from: {workspace_root}")
    print(f"Attempting to analyze folders: {', '.join(target_folders_to_analyze)}")

    all_results_found = False
    for folder_name in target_folders_to_analyze:
        current_target_directory = os.path.join(workspace_root, folder_name)

        if os.path.isdir(current_target_directory):
            print(f"\n--- Analyzing folder: {current_target_directory} ---")
            analyze_csv_files(current_target_directory)
            all_results_found = True
        else:
            print(f"Warning: Directory not found - {current_target_directory}. Skipping.")

    if not all_results_found and target_folders_to_analyze:
        print("\nNo specified target directories were found or processed.")
    elif not target_folders_to_analyze:
        print("\nNo target directories specified for analysis.")
    else:
        print("\n--- Analysis Complete ---")