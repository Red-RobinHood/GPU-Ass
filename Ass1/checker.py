import os
import re
import sys
import zipfile

def check_structure():
    cwd = os.getcwd()
    folder_name = os.path.basename(cwd)

    # 1. Check if folder name matches roll number format (UPPERCASE)
    roll_pattern = r"^[A-Z]{2}\d{2}[A-Z]+[0-9]+$"
    if not re.fullmatch(roll_pattern, folder_name):
        print(f"❌ Folder name '{folder_name}' is not in correct roll number format.")
        sys.exit(1)

    # 2. Required main.cu files
    required_main_files = [
        "main_1_1.cu",
        "main_1_2.cu",
        "main_2.cu",
        "main_3.cu",
        "main_4.cu"
    ]
    for f in required_main_files:
        if not os.path.isfile(f):
            print(f"❌ Missing file: {f}")
            sys.exit(1)

    # 3. Check report.pdf
    if not os.path.isfile("report.pdf"):
        print("❌ Missing file: report.pdf")
        sys.exit(1)

    # 4. Check public_test_cases folder
    if not os.path.isdir("public_test_cases"):
        print("❌ Missing folder: public_test_cases")
        sys.exit(1)

    # 5. Check contents of public_test_cases
    test_folder = "public_test_cases"
    files_in_test = os.listdir(test_folder)
    
    # At least one input CSV
    input_csvs = [f for f in files_in_test if f.endswith(".csv") and not f.startswith("output_")]
    if not input_csvs:
        print("❌ No input CSV file found in public_test_cases.")
        sys.exit(1)

    # Output files for each problem
    for prob in ["1_1", "1_2", "2", "3", "4"]:
        csv_name = f"output_{prob}_{folder_name}.csv"
        txt_name = f"output_{prob}_{folder_name}.txt"
        if csv_name not in files_in_test:
            print(f"❌ Missing output CSV: {csv_name}")
            sys.exit(1)
        if txt_name not in files_in_test:
            print(f"❌ Missing output TXT: {txt_name}")
            sys.exit(1)

    print("✅ Directory structure is correct.")

if __name__ == "__main__":
    check_structure()

