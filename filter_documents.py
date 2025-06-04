import pandas as pd
import os
import shutil
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def filter_documents(excel_path: str, source_folder: str) -> None:
    try:
        # Read the Excel file
        logger.info(f"Reading Excel file: {excel_path}")
        df = pd.read_excel(excel_path)
        
        # Get the first column values
        search_terms = df.iloc[:, 0].dropna().astype(str).tolist()
        logger.info(f"Found {len(search_terms)} search terms in Excel")
        
        # Create output folder
        output_folder = os.path.join(os.path.dirname(source_folder), 'filtered_documents')
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Created output folder: {output_folder}")
        
        # Search and copy matching files
        files_copied = 0
        for term in search_terms:
            for root, _, files in os.walk(source_folder):
                for file in files:
                    if term.lower() in file.lower():
                        source_path = os.path.join(root, file)
                        dest_path = os.path.join(output_folder, file)
                        shutil.copy2(source_path, dest_path)
                        logger.info(f"Copied: {file}")
                        files_copied += 1
        
        logger.info(f"Process completed. {files_copied} files copied to {output_folder}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python filter_documents.py <excel_file_path> <source_folder_path>")
        sys.exit(1)
    
    excel_path = sys.argv[1]
    source_folder = sys.argv[2]
    
    # Validate inputs
    if not os.path.exists(excel_path):
        logger.error(f"Excel file not found: {excel_path}")
        sys.exit(1)
    if not os.path.exists(source_folder):
        logger.error(f"Source folder not found: {source_folder}")
        sys.exit(1)
    
    filter_documents(excel_path, source_folder)

if __name__ == "__main__":
    main() 