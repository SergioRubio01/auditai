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

import boto3
from botocore.exceptions import ClientError
from typing import Dict, Any, Tuple, List
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

class TextractProcessor:
    """
    Class TextractProcessor uses Amazon Textract OCR tool to process tables of invoices and payments.
    A starting point to use it is:
    ```python
    tp = TextractProcessor()
    ```
    """
    def __init__(self):
        # Get AWS credentials and region from environment variables
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION', 'eu-west-1')

        if not all([aws_access_key, aws_secret_key, aws_region]):
            logger.warning("AWS credentials or region not fully configured")
            raise ValueError(
                "Missing AWS credentials. Please ensure AWS_ACCESS_KEY_ID, "
                "AWS_SECRET_ACCESS_KEY, and AWS_REGION are set in your .env file"
            )

        self.client = boto3.client(
            'textract',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region,
            endpoint_url=f'https://textract.{aws_region}.amazonaws.com'
        )
        
        self.confidence_threshold = float(os.getenv('TEXTRACT_CONFIDENCE_THRESHOLD', '0.7'))

    async def process_document_tables(
        self, 
        file_bytes: bytes,
    ) -> Dict[str, Any]:
        """
        Process document with Textract focusing only on tables
        
        Args:
            file_bytes: Document bytes
        """
        try:
            # Base request parameters with only TABLES feature
            request_params = {
                'Document': {'Bytes': file_bytes},
                'FeatureTypes': ['TABLES']
            }
            
            response = self.client.analyze_document(**request_params)
            
            if 'Blocks' not in response:
                logger.error(f"Unexpected Textract response: {response}")
                raise ValueError("Invalid Textract response format")
            
            # Extract only tables
            extracted_data = {
                'tables': self._extract_tables(response['Blocks'])
            }
            
            # confidence = self._calculate_confidence(response['Blocks'])
            
            return extracted_data
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"AWS Textract error: {error_code} - {error_message}")
            raise
        except Exception as e:
            logger.error(f"Textract processing error: {str(e)}")
            raise

    def _extract_tables(self, blocks: list) -> List[Dict[str, Any]]:
        """
        Extract only tables from Textract blocks
        """
        tables = []
        current_table = None
        
        try:
            # First pass: identify tables and create their structure
            for block in blocks:
                if block['BlockType'] == 'TABLE':
                    current_table = {
                        'cells': [],
                        # 'confidence': block.get('Confidence', 0),
                        # 'geometry': block.get('Geometry', {}),
                        'id': block.get('Id', '')
                    }
                    tables.append(current_table)
                    
                elif block['BlockType'] == 'CELL' and current_table is not None:
                    row_index = block.get('RowIndex', 1) - 1
                    col_index = block.get('ColumnIndex', 1) - 1
                    
                    # Ensure we have enough rows
                    while len(current_table['cells']) <= row_index:
                        current_table['cells'].append([])
                        
                    # Ensure we have enough columns in the current row
                    while len(current_table['cells'][row_index]) <= col_index:
                        current_table['cells'][row_index].append('')
                
                    # Get cell text
                    cell_text = ''
                    if 'Relationships' in block:
                        for relationship in block['Relationships']:
                            if relationship['Type'] == 'CHILD':
                                for child_id in relationship['Ids']:
                                    child_block = next(
                                        (b for b in blocks if b['Id'] == child_id), 
                                        None
                                    )
                                    if child_block and child_block['BlockType'] == 'WORD':
                                        cell_text += child_block.get('Text', '') + ' '
                
                    # Store cell information
                    current_table['cells'][row_index][col_index] = {
                        'text': cell_text.strip(),
                        # 'confidence': block.get('Confidence', 0),
                        # 'rowspan': block.get('RowSpan', 1),
                        # 'colspan': block.get('ColumnSpan', 1),
                        # 'geometry': block.get('Geometry', {}),
                        'is_header': 'COLUMN_HEADER' in block.get('EntityTypes', [])
                    }
            
            # Post-process tables
            processed_tables = []
            for table in tables:
                if table['cells']:  # Only include tables that have cells
                    # Calculate table-level confidence
                    # cell_confidences = [
                    #     cell['confidence']
                    #     for row in table['cells']
                    #     for cell in row
                    #     if isinstance(cell, dict)
                    # ]
                    # table['confidence'] = sum(cell_confidences) / len(cell_confidences) if cell_confidences else 0
                    
                    # Clean up empty cells
                    table['cells'] = [
                        [cell if isinstance(cell, dict) else {'text': ''} for cell in row]
                        for row in table['cells']
                    ]
                    
                    processed_tables.append(table)
            
            return processed_tables
            
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
            return []

    def _calculate_confidence(self, blocks: list) -> float:
        """Calculate average confidence score from blocks"""
        confidences = []
        for block in blocks:
            if 'Confidence' in block:
                confidences.append(block['Confidence'])
        return sum(confidences) / len(confidences) if confidences else 0.0

    def test_connection_sync(self) -> bool:
        """Test the Textract connection synchronously"""
        try:
            self.client.describe_document_analysis()
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'InvalidRequest':
                return True
            logger.error(f"Textract connection test failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Textract connection test failed: {str(e)}")
            return False

    async def test_connection(self) -> bool:
        """Test the Textract connection asynchronously"""
        return self.test_connection_sync()