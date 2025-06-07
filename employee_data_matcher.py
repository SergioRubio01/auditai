"""
Employee Data Matcher
-------------------
This script matches and corrects employee data between two Excel sheets
using a hybrid approach of traditional matching and LLM verification.
Optimized for cloud deployment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import os
from openai import OpenAI
from dotenv import load_dotenv
from thefuzz import fuzz
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import numpy as np
from typing import Dict, List, Tuple, Set
import multiprocessing
import boto3
from botocore.exceptions import ClientError
import tempfile

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='employee_matcher.log'
)
logger = logging.getLogger(__name__)

# Cloud configuration
S3_BUCKET = os.getenv('S3_BUCKET', 'employee-matcher')
AWS_REGION = os.getenv('AWS_REGION', 'eu-west-3')

@dataclass
class MatchResult:
    """Class to store matching results for an employee."""
    original_trabajador: str
    original_nif: str
    matched_trabajador: str
    matched_nif: str
    confidence_score: float
    row_index: int
    match_explanation: str

def get_optimal_batch_size():
    """Determine optimal batch size based on available CPU cores and memory."""
    cpu_count = multiprocessing.cpu_count()
    total_memory_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3)
    
    # Adjust batch size based on available resources
    if total_memory_gb > 30:  # High-memory instance
        return 200
    elif total_memory_gb > 15:  # Medium-memory instance
        return 100
    else:  # Low-memory instance
        return 50

def upload_to_s3(file_path: str, bucket: str, key: str) -> bool:
    """Upload a file to S3."""
    try:
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        s3_client.upload_file(file_path, bucket, key)
        return True
    except ClientError as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        return False

def download_from_s3(bucket: str, key: str, local_path: str) -> bool:
    """Download a file from S3."""
    try:
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        s3_client.download_file(bucket, key, local_path)
        return True
    except ClientError as e:
        logger.error(f"Error downloading from S3: {str(e)}")
        return False

def clean_nif(nif: str) -> str:
    """Clean and standardize NIF format."""
    if pd.isna(nif):
        return ""
    nif = str(nif).upper().strip()
    return re.sub(r'[^A-Z0-9]', '', nif)

def clean_name(name: str) -> str:
    """Clean and standardize name format."""
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()
    name = re.sub(r'\s+', ' ', name)
    return re.sub(r'[^\w\s]', '', name)

def preprocess_dataframe(df: pd.DataFrame, nif_col: str, name_col: str = 'Trabajador') -> pd.DataFrame:
    """Preprocess an entire dataframe at once."""
    df = df.copy()
    df['clean_nif'] = df[nif_col].apply(clean_nif)
    df['clean_name'] = df[name_col].apply(clean_name)
    df['name_tokens'] = df['clean_name'].str.split()
    return df

def calculate_similarity_matrix(nominas_batch: pd.DataFrame, cj_df: pd.DataFrame) -> np.ndarray:
    """Calculate similarity matrix for all pairs at once."""
    n_nominas = len(nominas_batch)
    n_cj = len(cj_df)
    
    # Initialize similarity matrices
    ratio_matrix = np.zeros((n_nominas, n_cj))
    partial_ratio_matrix = np.zeros((n_nominas, n_cj))
    token_sort_matrix = np.zeros((n_nominas, n_cj))
    nif_match_matrix = np.zeros((n_nominas, n_cj))
    
    # Vectorized NIF matching
    nif_match_matrix = (nominas_batch['clean_nif'].values[:, None] == cj_df['clean_nif'].values).astype(float)
    
    # Calculate other similarity metrics using parallel processing
    def calc_row_similarities(i, row):
        for j, cj_row in enumerate(cj_df['clean_name']):
            ratio_matrix[i, j] = fuzz.ratio(row, cj_row) / 100
            partial_ratio_matrix[i, j] = fuzz.partial_ratio(row, cj_row) / 100
            token_sort_matrix[i, j] = fuzz.token_sort_ratio(row, cj_row) / 100
    
    with ThreadPoolExecutor() as executor:
        list(executor.map(
            lambda x: calc_row_similarities(x[0], x[1]),
            enumerate(nominas_batch['clean_name'])
        ))
    
    # Combine matrices with weights
    similarity_matrix = (
        0.3 * ratio_matrix +
        0.3 * partial_ratio_matrix +
        0.2 * token_sort_matrix +
        0.2 * nif_match_matrix
    )
    
    return similarity_matrix

def calculate_similarity_score(row1: pd.Series, row2: pd.Series) -> float:
    """Calculate similarity score between two rows using multiple metrics."""
    name1 = clean_name(row1['Trabajador'])
    name2 = clean_name(row2['Trabajador'])
    nif1 = clean_nif(row1['NIF'])
    nif2 = clean_nif(row2['DNI'])
    
    # Calculate name similarity using different metrics
    name_ratio = fuzz.ratio(name1, name2) / 100
    name_partial_ratio = fuzz.partial_ratio(name1, name2) / 100
    name_token_sort_ratio = fuzz.token_sort_ratio(name1, name2) / 100
    
    # NIF similarity
    nif_match = 1.0 if nif1 and nif2 and nif1 == nif2 else 0.0
    
    # Weight the scores (adjust weights based on importance)
    weighted_score = (
        0.3 * name_ratio +
        0.3 * name_partial_ratio +
        0.2 * name_token_sort_ratio +
        0.2 * nif_match
    )
    
    return weighted_score

def verify_matches_with_llm(client: OpenAI, matches: List[Tuple[pd.Series, pd.Series, float]], batch_size: int = 20) -> List[Tuple[float, str]]:
    """Verify a batch of matches using LLM."""
    try:
        # Prepare the prompt with all matches
        matches_text = ""
        for i, (row1, row2, _) in enumerate(matches, 1):
            matches_text += f"\nPair {i}:\nRecord 1:\n- Name: {row1['Trabajador']}\n- NIF: {row1['NIF']}\n\nRecord 2:\n- Name: {row2['Trabajador']}\n- NIF: {row2['DNI']}\n"

        prompt = f"""You are an expert in matching Spanish names and identification numbers.
Analyze these {len(matches)} pairs of employee records and determine if each pair refers to the same person.
Consider name variations, Spanish naming conventions, typos, and NIF format.

{matches_text}

For each pair, respond with a line in this format:
PAIR_NUMBER|CONFIDENCE_SCORE|EXPLANATION
Example: "1|0.95|Names match perfectly with slight variation in order, NIFs match exactly"

Respond with exactly {len(matches)} lines, one for each pair."""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a Spanish employee data matching expert. Respond only with the pair number, score, and explanation in the exact format requested."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        # Parse results
        results = []
        for line in response.choices[0].message.content.strip().split('\n'):
            pair_num, score, explanation = line.split('|', 2)
            results.append((float(score), explanation))
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch LLM verification: {str(e)}")
        return [(0.0, f"Error: {str(e)}")] * len(matches)

def process_batch_efficiently(args):
    """Process a batch of rows using efficient matching strategy."""
    nominas_batch, cj_df, threshold, client = args
    matches = []
    unmatched = []
    
    # Preprocess both dataframes
    nominas_batch = preprocess_dataframe(nominas_batch, 'NIF')
    
    # Calculate similarity matrix for all pairs at once
    similarity_matrix = calculate_similarity_matrix(nominas_batch, cj_df)
    
    # Process all rows in batch using vectorized operations
    initial_threshold = 0.6
    batch_size_llm = 20  # Process LLM verifications in sub-batches
    
    # Find potential matches for all rows at once
    potential_matches_all = []
    original_indices = []
    
    for idx in range(len(nominas_batch)):
        row_scores = similarity_matrix[idx]
        top_indices = np.where(row_scores >= initial_threshold)[0]
        
        if len(top_indices) > 0:
            # Sort by score and take top 3
            top_scores = row_scores[top_indices]
            top_k = min(3, len(top_indices))
            top_k_indices = top_indices[np.argsort(top_scores)[-top_k:]]
            
            for j in top_k_indices:
                potential_matches_all.append((
                    nominas_batch.iloc[idx],
                    cj_df.iloc[j],
                    row_scores[j]
                ))
                original_indices.append(idx)
    
    # Process potential matches in batches to avoid API rate limits
    for i in range(0, len(potential_matches_all), batch_size_llm):
        batch_matches = potential_matches_all[i:i + batch_size_llm]
        batch_indices = original_indices[i:i + batch_size_llm]
        
        if not batch_matches:
            continue
        
        # Verify batch with LLM
        llm_results = verify_matches_with_llm(client, batch_matches)
        
        # Process LLM results
        for (score, explanation), (row, cj_row, _), orig_idx in zip(llm_results, batch_matches, batch_indices):
            if score >= threshold:
                matches.append(MatchResult(
                    original_trabajador=str(row['Trabajador']),
                    original_nif=str(row['NIF']),
                    matched_trabajador=str(cj_row['Trabajador']),
                    matched_nif=str(cj_row['DNI']),
                    confidence_score=score,
                    row_index=nominas_batch.index[orig_idx],
                    match_explanation=explanation
                ))
                # Remove this index from further consideration
                original_indices = [idx for idx in original_indices if idx != orig_idx]
                break  # Stop checking other potential matches for this row
    
    # Add unmatched rows
    processed_indices = set(m.row_index for m in matches)
    for idx in nominas_batch.index:
        if idx not in processed_indices:
            unmatched.append({
                'row_index': idx,
                'trabajador': str(nominas_batch.loc[idx, 'Trabajador']),
                'nif': str(nominas_batch.loc[idx, 'NIF']),
                'reason': 'No high-confidence match found'
            })
    
    return matches, unmatched

class EmployeeMatcher:
    def __init__(self, excel_path: str, similarity_threshold: float = 0.7):
        """Initialize the matcher with Excel file path and similarity threshold."""
        self.excel_path = Path(excel_path)
        self.similarity_threshold = similarity_threshold
        self.nominas_df = None
        self.cj_df = None
        self.matches: List[MatchResult] = []
        self.unmatched_rows: List[Dict] = []
        self.client = OpenAI()
        
    def load_data(self) -> None:
        """Load data from Excel sheets."""
        try:
            logger.info(f"Reading Excel file: {self.excel_path}")
            self.nominas_df = pd.read_excel(self.excel_path, sheet_name="Nominas Desglose")
            self.cj_df = pd.read_excel(self.excel_path, sheet_name="CJ Sergio")
            
            # Preprocess CJ dataframe once since it's used for all comparisons
            self.cj_df = preprocess_dataframe(self.cj_df, 'DNI')
            
            logger.info("Successfully loaded and preprocessed both sheets")
            logger.info(f"Nominas Desglose columns: {list(self.nominas_df.columns)}")
            logger.info(f"CJ columns: {list(self.cj_df.columns)}")
            
        except Exception as e:
            logger.error(f"Error loading Excel sheets: {str(e)}")
            raise

    def process_rows(self) -> None:
        """Process rows that need correction."""
        try:
            # Filter rows where EN CJ is not null
            rows_to_process = self.nominas_df[pd.isna(self.nominas_df['EN CJ'])]
            logger.info(f"Found {len(rows_to_process)} rows to process")
            
            # Process in larger batches for better efficiency
            batch_size = 100  # Increased batch size for better vectorization
            batches = [rows_to_process[i:i + batch_size] 
                      for i in range(0, len(rows_to_process), batch_size)]
            
            # Process batches with progress bar
            with tqdm(total=len(batches), desc="Processing batches") as pbar:
                for batch in batches:
                    matches, unmatched = process_batch_efficiently(
                        (batch, self.cj_df, self.similarity_threshold, self.client)
                    )
                    self.matches.extend(matches)
                    self.unmatched_rows.extend(unmatched)
                    pbar.update(1)
            
            logger.info(f"Processed {len(self.matches)} successful matches")
            logger.info(f"Found {len(self.unmatched_rows)} unmatched rows")
            
        except Exception as e:
            logger.error(f"Error processing rows: {str(e)}")
            raise

    def apply_corrections(self) -> None:
        """Apply corrections to the Excel file."""
        try:
            updates = 0
            # Prepare updates as a batch
            update_data = {
                'Trabajador': pd.Series({m.row_index: m.matched_trabajador for m in self.matches}),
                'NIF': pd.Series({m.row_index: m.matched_nif for m in self.matches}),
                'EN CJ': pd.Series({m.row_index: 'SI' for m in self.matches}),
                'Match Confidence': pd.Series({m.row_index: m.confidence_score for m in self.matches}),
                'Match Explanation': pd.Series({m.row_index: m.match_explanation for m in self.matches})
            }
            
            # Add new columns if they don't exist
            for col in ['Match Confidence', 'Match Explanation']:
                if col not in self.nominas_df.columns:
                    self.nominas_df[col] = None
            
            # Apply updates in batch
            for col, values in update_data.items():
                self.nominas_df.loc[values.index, col] = values
                updates += len(values)
            
            # Save the updated file
            with pd.ExcelWriter(self.excel_path, engine='openpyxl', mode='a', 
                              if_sheet_exists='replace') as writer:
                self.nominas_df.to_excel(writer, sheet_name='Nominas Desglose', index=False)
            
            logger.info(f"Successfully applied {updates} corrections")
            
        except Exception as e:
            logger.error(f"Error applying corrections: {str(e)}")
            raise

    def generate_report(self) -> Dict:
        """Generate a detailed report of changes made."""
        report = {
            'summary': {
                'total_processed': len(self.matches) + len(self.unmatched_rows),
                'successful_matches': len(self.matches),
                'unmatched_rows': len(self.unmatched_rows),
                'timestamp': datetime.now().isoformat()
            },
            'corrected_rows': [
                {
                    'row_index': m.row_index,
                    'original': {
                        'trabajador': m.original_trabajador,
                        'nif': m.original_nif
                    },
                    'corrected': {
                        'trabajador': m.matched_trabajador,
                        'nif': m.matched_nif
                    },
                    'confidence_score': m.confidence_score,
                    'explanation': m.match_explanation
                } for m in self.matches
            ],
            'unmatched_rows': self.unmatched_rows,
            'recommendations': [
                "Review unmatched rows manually",
                "Check high-confidence matches first",
                "Consider reviewing matches with confidence scores below 0.85",
                "Verify corrections in the Excel file"
            ]
        }
        
        # Save report to file
        report_path = self.excel_path.parent / 'employee_matcher_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report

class CloudEmployeeMatcher(EmployeeMatcher):
    def __init__(self, excel_path: str, similarity_threshold: float = 0.7):
        """Initialize the matcher with cloud optimization."""
        # Fix S3 path format if needed
        if excel_path.startswith('s3:/') and not excel_path.startswith('s3://'):
            excel_path = 's3://' + excel_path[4:]  # Convert s3:/ to s3://
            
        super().__init__(excel_path, similarity_threshold)
        self.batch_size = get_optimal_batch_size()
        self.temp_dir = tempfile.mkdtemp()
        
    def load_data(self) -> None:
        """Load data with cloud optimization."""
        try:
            logger.info(f"Reading Excel file: {self.excel_path}")
            
            # Convert Path to string for S3 check
            excel_path_str = str(self.excel_path)
            
            # First check if it's an S3 path
            if excel_path_str.startswith('s3:/') or excel_path_str.startswith('s3://'):
                try:
                    # Normalize S3 path format
                    if excel_path_str.startswith('s3:/'):
                        excel_path_str = 's3://' + excel_path_str[4:]
                    
                    # Parse S3 path correctly
                    s3_parts = excel_path_str[5:].split('/')  # Skip 's3://'
                    if not s3_parts:
                        raise ValueError("Invalid S3 path format")
                        
                    bucket = s3_parts[0]
                    key = '/'.join(s3_parts[1:]) if len(s3_parts) > 1 else ''
                    
                    if not bucket or not key:
                        raise ValueError(f"Invalid S3 path: bucket={bucket}, key={key}")
                    
                    logger.info(f"Downloading from S3 bucket: {bucket}, key: {key}")
                    local_path = os.path.join(self.temp_dir, 'input.xlsx')
                    
                    # Initialize S3 client
                    s3_client = boto3.client('s3', region_name=AWS_REGION)
                    
                    # Download file
                    try:
                        s3_client.download_file(bucket, key, local_path)
                        logger.info(f"Successfully downloaded file to {local_path}")
                    except ClientError as e:
                        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                        if error_code == 'NoSuchBucket':
                            raise Exception(f"S3 bucket '{bucket}' does not exist")
                        elif error_code == 'NoSuchKey':
                            raise Exception(f"File '{key}' not found in bucket '{bucket}'")
                        else:
                            raise Exception(f"S3 error: {str(e)}")
                    
                    self.excel_path = Path(local_path)
                except Exception as s3_error:
                    logger.error(f"Error downloading from S3: {str(s3_error)}")
                    raise Exception(f"Failed to download file from S3: {str(s3_error)}")
            else:
                # Only check local file if it's not an S3 path
                if not os.path.exists(self.excel_path):
                    raise FileNotFoundError(f"Local file not found: {self.excel_path}")
            
            # Load and preprocess data
            logger.info(f"Reading Excel file from: {self.excel_path}")
            self.nominas_df = pd.read_excel(self.excel_path, sheet_name="Nominas Desglose")
            self.cj_df = pd.read_excel(self.excel_path, sheet_name="CJ Sergio")
            
            # Preprocess CJ dataframe once
            self.cj_df = preprocess_dataframe(self.cj_df, 'DNI')
            
            # Optional: Cache preprocessed data in memory
            if os.getenv('CACHE_IN_MEMORY', 'true').lower() == 'true':
                self.cj_df = self.cj_df.copy()
            
            logger.info("Successfully loaded and preprocessed both sheets")
            logger.info(f"Nominas Desglose columns: {list(self.nominas_df.columns)}")
            logger.info(f"CJ columns: {list(self.cj_df.columns)}")
            
        except Exception as e:
            logger.error(f"Error loading Excel sheets: {str(e)}")
            raise

    def process_rows(self) -> None:
        """Process rows with cloud optimization."""
        try:
            rows_to_process = self.nominas_df[pd.isna(self.nominas_df['EN CJ'])]
            total_rows = len(rows_to_process)
            logger.info(f"Found {total_rows} rows to process")
            
            # Determine optimal processing strategy based on data size and available resources
            cpu_count = multiprocessing.cpu_count()
            if total_rows > 10000 and cpu_count > 1:  # Large dataset and multiple CPUs
                self._process_large_dataset(rows_to_process)
            else:  # Small to medium dataset or single CPU
                self._process_medium_dataset(rows_to_process)
            
        except Exception as e:
            logger.error(f"Error processing rows: {str(e)}")
            raise

    def _process_large_dataset(self, rows_to_process: pd.DataFrame) -> None:
        """Process large datasets using distributed processing."""
        try:
            cpu_count = multiprocessing.cpu_count()
            # Ensure we don't create too many chunks
            chunk_size = max(1000, len(rows_to_process) // min(cpu_count * 2, 8))
            
            # Split data into chunks
            chunks = [rows_to_process[i:i + chunk_size] 
                     for i in range(0, len(rows_to_process), chunk_size)]
            
            logger.info(f"Processing {len(chunks)} chunks using {cpu_count} CPUs")
            
            # Process chunks in parallel
            with ProcessPoolExecutor(max_workers=cpu_count) as executor:
                futures = []
                for chunk in chunks:
                    future = executor.submit(
                        process_batch_efficiently,
                        (chunk, self.cj_df, self.similarity_threshold, self.client)
                    )
                    futures.append(future)
                
                # Collect results with progress bar
                with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
                    for future in futures:
                        try:
                            matches, unmatched = future.result()
                            self.matches.extend(matches)
                            self.unmatched_rows.extend(unmatched)
                        except Exception as e:
                            logger.error(f"Error processing chunk: {str(e)}")
                        pbar.update(1)
                
        except Exception as e:
            logger.error(f"Error in large dataset processing: {str(e)}")
            raise

    def _process_medium_dataset(self, rows_to_process: pd.DataFrame) -> None:
        """Process medium-sized datasets with optimized batching."""
        batches = [rows_to_process[i:i + self.batch_size] 
                  for i in range(0, len(rows_to_process), self.batch_size)]
        
        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            for batch in batches:
                matches, unmatched = process_batch_efficiently(
                    (batch, self.cj_df, self.similarity_threshold, self.client)
                )
                self.matches.extend(matches)
                self.unmatched_rows.extend(unmatched)
                pbar.update(1)

    def save_results(self) -> None:
        """Save results with cloud storage support."""
        try:
            # Save locally first
            super().apply_corrections()
            
            # Upload to S3 if configured
            if os.getenv('USE_S3_STORAGE', 'false').lower() == 'true':
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Upload updated Excel file
                excel_key = f'results/TOTAL_MESES_CDI_{timestamp}.xlsx'
                upload_to_s3(str(self.excel_path), S3_BUCKET, excel_key)
                
                # Upload report
                report = self.generate_report()
                report_path = os.path.join(self.temp_dir, f'report_{timestamp}.json')
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                report_key = f'reports/employee_matcher_report_{timestamp}.json'
                upload_to_s3(report_path, S3_BUCKET, report_key)
                
                logger.info(f"Results uploaded to S3: {S3_BUCKET}/{excel_key}")
                
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

def main():
    """Main entry point for the script."""
    try:
        print("Starting employee data matching process...")
        start_time = datetime.now()
        
        # Get configuration from environment
        excel_path = os.getenv('EXCEL_PATH', "TOTAL MESES CDI.xlsx")
        
        # Fix S3 path format if needed
        if excel_path.startswith('s3:/') and not excel_path.startswith('s3://'):
            excel_path = 's3://' + excel_path[4:]  # Convert s3:/ to s3://
        
        similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.75'))
        
        # Initialize cloud-optimized matcher
        matcher = CloudEmployeeMatcher(excel_path, similarity_threshold)
        
        # Execute matching process
        matcher.load_data()
        matcher.process_rows()
        matcher.save_results()
        
        # Generate and print report
        report = matcher.generate_report()
        end_time = datetime.now()
        
        print("\nEmployee Data Matching Report")
        print("============================")
        print(f"Total rows processed: {report['summary']['total_processed']}")
        print(f"Successful matches: {report['summary']['successful_matches']}")
        print(f"Unmatched rows: {report['summary']['unmatched_rows']}")
        print(f"\nProcessing time: {end_time - start_time}")
        print("\nDetailed report saved to: employee_matcher_report.json")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"An error occurred. Please check the log file for details: {str(e)}")
        raise  # Re-raise the exception for proper error handling

if __name__ == "__main__":
    main() 