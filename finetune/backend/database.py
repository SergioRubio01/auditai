# Copyright 2025-present Sergio García Arrojo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Dataset Creation and Management System

This script handles the creation and organization of training datasets,
specifically for document processing tasks. It combines image data with
corresponding Excel annotations to create JSONL training files.

Key components:
- Excel Processing: Reads and validates annotation data from multiple sheets
- Image Management: Matches images with corresponding annotations
- JSONL Creation: Builds training data in JSONL format for fine-tuning
- Data Validation: Ensures data integrity and completeness

Key functions:
- create_dataset(): Main function for dataset creation
- encode_example(): Processes individual data examples
- convert_to_jsonl(): Converts data to JSONL training format
- Command-line interface for dataset creation

Usage:
    python database.py -c <community_name> -t <source_type>

Arguments:
    -c, --comunidad: Name of the community folder to process
    -t, --type: Source type (either 'excel' or 'json')

Directory Structure:
    <community_name>/
        ├── PagoImages/    # Folder containing document images
        ├── dataset/       # Output folder for created dataset
        │   └── train.jsonl # JSONL training data file
        └── <excel_file>   # Excel file with annotations

Output Format:
    JSONL file with entries containing:
    - Base64 encoded image data
    - Raw text data from Excel sheets
    - Formatted conversation data for training

Error Handling:
    - Validates image existence and format
    - Handles missing or invalid Excel data
    - Reports processing errors for individual entries
    - Validates JSONL output format
"""

import os
import pandas as pd
from datasets import Dataset
from PIL import Image
import argparse
import json
import base64
from io import BytesIO

def create_dataset(comunidad: str, excel_filename: str, sheet_name: str):
    # Define paths
    image_folder = f"{comunidad}/Images"
    dataframe_path = f"{comunidad}/{excel_filename}"

    # Load the pandas DataFrame
    df = pd.read_excel(dataframe_path, sheet_name=sheet_name)

    # Initialize lists for image paths and text data
    image_paths = []
    raw_data = []

    # Iterate through the DataFrame to match images with text
    for _, row in df.iterrows():
        image_file = os.path.join(image_folder, row["FICHERO"] + ".png")
        if not os.path.exists(image_file):  # Skip if image doesn't exist
            print(f"Skipping {image_file} because it doesn't exist")
            continue

        # Get all columns as raw data strings
        raw_data_strings = []
        for col in df.columns:
            if pd.notna(row[col]):
                raw_data_strings.append(str(row[col]))

        image_paths.append(image_file)
        raw_data.append(raw_data_strings)

    # Create DataFrame only when we have matching data
    dataset_df = pd.DataFrame({
        "image": image_paths,
        "raw_data": raw_data
    })

    def encode_example(row):
        try:
            image_path = row["image"]
            with Image.open(image_path).convert("RGB") as img:
                img_buffer = BytesIO()
                img.save(img_buffer, format="PNG")
                img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error loading image {row['image']}: {e}")
            img_str = None
        return {
            "image": img_str,
            "raw_data": row["raw_data"]
        }

    # Convert the DataFrame into a Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(dataset_df)
    hf_dataset = hf_dataset.map(encode_example, batched=False)

    # Save the dataset
    hf_dataset.save_to_disk(comunidad + "/dataset")
    return hf_dataset

def create_jsonl(comunidad: str):
    # Load the dataset
    hf_dataset = Dataset.load_from_disk(comunidad + "/dataset")
    jsonl_path = os.path.join(comunidad, "dataset.jsonl")
    output = []

    for idx, row in enumerate(hf_dataset):
        try:
            if row["image"] is not None:
                new_entry = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that extracts data from images of invoices and provides the data in a structured format."},
                        {"role": "user", "content": "All available information is in the image. Please extract the information from the image."},
                        {"role": "user", "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{row['image']}"
                                }
                            }
                        ]},
                        {"role": "assistant", "content": json.dumps(row["raw_data"], ensure_ascii=False)},
                    ]
                }

                # Validate JSON before adding
                json.dumps(new_entry, ensure_ascii=False)
                output.append(new_entry)

                # Debug first few entries
                if idx < 3:
                    print(f"\nEntry {idx} structure:")
                    print(json.dumps(new_entry["messages"], indent=2)[:200] + "...")

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    # Write to JSONL file
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create dataset from images and Excel file')
    parser.add_argument('-c', '--comunidad', type=str, required=True, help='Name of the Comunidad')
    parser.add_argument('-e', '--excel', type=str, required=True, help='Name of the Excel file')
    parser.add_argument('-s', '--sheet', type=str, required=True, help='Name of the sheet in the Excel file')

    args = parser.parse_args()

    print(f"Creating dataset for {args.comunidad}")
    hf_dataset = create_dataset(args.comunidad, args.excel, args.sheet)
    print(hf_dataset)

    print("Creating JSONL file")
    create_jsonl(args.comunidad)
    print("Done")
