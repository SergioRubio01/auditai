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
PagosICEX is a Python package that provides an AI-powered document processing system for financial transactions. It uses computer vision and natural language processing to analyze and extract information from financial documents, with specialized handling for bank transfers and card transactions. The package includes a modular agent-based architecture for document classification, information retrieval, and data processing.

Key features:
- Document type classification using vision-language models
- Specialized agents for processing transfers and card transactions 
- Modular and extensible agent architecture
- Integration with unsloth for optimized model inference
- Comprehensive logging and error handling
"""
from pathlib import Path
from dotenv import load_dotenv
import os
import logging
from typing import List, Optional

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables from multiple possible locations."""
    env_locations = [
        Path.cwd() / '.env',                    # Current working directory
        Path(__file__).parent.parent / '.env',  # Project root
        Path.home() / '.env'                    # User's home directory
    ]
    
    for env_path in env_locations:
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=True)
            logger.info(f"Loaded environment variables from {env_path}")
            return True
    
    logger.warning("No .env file found in standard locations")
    return False

# Initialize environment BEFORE any other imports
if not load_environment():
    raise EnvironmentError("No environment file found")

# Verify critical environment variables
required_vars = ['OPENAI_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Package metadata
__version__ = "0.1.0"
__author__ = "Sergio García Arrojo"
# Now that environment is loaded, we can import components
from .models import *  # Consider explicit imports
from .routes import *  # Consider explicit imports
from .processor import process_single_image

__all__ = [
    "process_single_image"
]