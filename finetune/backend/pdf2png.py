"""
PDF to PNG Converter for Document Processing with Large File Support

This script processes PDF documents for a document management system, with special
handling for large PDFs to prevent memory issues. Key features:
- Splits large PDFs into smaller chunks (100 pages max)
- Memory-efficient processing
- Progress tracking
- Error handling
"""

import os
from pdf2image import convert_from_path
import pandas as pd
import argparse
from PIL import Image
import shutil
import PyPDF2
import logging
from typing import List, Tuple
import math
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_PAGES_PER_CHUNK = 100
MEMORY_LIMIT_MB = 1024  # 1GB memory limit

def split_pdf(input_path: str, output_dir: str, pages_per_chunk: int = MAX_PAGES_PER_CHUNK) -> List[str]:
    """
    Split a large PDF into smaller chunks.
    
    Args:
        input_path: Path to input PDF file
        output_dir: Directory to save split PDFs
        pages_per_chunk: Maximum pages per split PDF
        
    Returns:
        List of paths to split PDF files
    """
    try:
        with open(input_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            total_pages = len(pdf.pages)
            
            if total_pages <= pages_per_chunk:
                return [input_path]
                
            logger.info(f"Splitting PDF with {total_pages} pages into chunks of {pages_per_chunk} pages")
            
            split_pdfs = []
            num_chunks = math.ceil(total_pages / pages_per_chunk)
            
            for chunk in range(num_chunks):
                start_page = chunk * pages_per_chunk
                end_page = min((chunk + 1) * pages_per_chunk, total_pages)
                
                pdf_writer = PyPDF2.PdfWriter()
                
                # Add pages for this chunk
                for page_num in range(start_page, end_page):
                    pdf_writer.add_page(pdf.pages[page_num])
                
                # Generate output filename
                base_name = Path(input_path).stem
                chunk_filename = f"{base_name}_part{chunk + 1}.pdf"
                output_path = os.path.join(output_dir, chunk_filename)
                
                # Save the chunk
                with open(output_path, 'wb') as output_file:
                    pdf_writer.write(output_file)
                
                split_pdfs.append(output_path)
                logger.info(f"Created chunk {chunk + 1}/{num_chunks}: {chunk_filename}")
            
            return split_pdfs
            
    except Exception as e:
        logger.error(f"Error splitting PDF {input_path}: {str(e)}")
        raise

def create_single_image(images: List[Image.Image]) -> Image.Image:
    """Creates a single image from multiple PIL images by stacking them vertically."""
    if not images:
        raise ValueError("No images provided")
        
    total_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)
    
    # Check memory requirements
    estimated_memory = (total_width * total_height * 3) / (1024 * 1024)  # in MB
    if estimated_memory > MEMORY_LIMIT_MB:
        raise MemoryError(f"Combined image would require {estimated_memory:.2f}MB, exceeding {MEMORY_LIMIT_MB}MB limit")
    
    combined_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height
    
    return combined_image

def convert_pdf_chunk(pdf_path: str, output_dir: str, base_filename: str, start_page: int = 0) -> List[str]:
    """
    Convert a PDF chunk to PNG images.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save PNG files
        base_filename: Base name for output files
        start_page: Starting page number for filename continuation
        
    Returns:
        List of paths to generated PNG files
    """
    try:
        images = convert_from_path(pdf_path)
        output_paths = []
        
        for i, image in enumerate(images, start_page + 1):
            output_filename = f"{base_filename}({i}).png"
            output_path = os.path.join(output_dir, output_filename)
            image.save(output_path, "PNG")
            output_paths.append(output_path)
            logger.info(f"Created {output_filename}")
            
        return output_paths
        
    except Exception as e:
        logger.error(f"Error converting PDF chunk {pdf_path}: {str(e)}")
        raise

def process_pdf(pdf_path: str, output_dir: str, temp_dir: str) -> List[str]:
    """
    Process a single PDF file, handling splitting if necessary.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save PNG files
        temp_dir: Directory for temporary split PDFs
        
    Returns:
        List of paths to generated PNG files
    """
    try:
        # Create temporary directory for PDF chunks if needed
        os.makedirs(temp_dir, exist_ok=True)
        
        # Split PDF if necessary
        pdf_chunks = split_pdf(pdf_path, temp_dir)
        base_filename = Path(pdf_path).stem
        
        # Process each chunk
        png_files = []
        current_page = 0
        
        for chunk_path in pdf_chunks:
            chunk_pngs = convert_pdf_chunk(
                chunk_path, 
                output_dir, 
                base_filename, 
                current_page
            )
            png_files.extend(chunk_pngs)
            current_page += len(chunk_pngs)
            
            # Clean up temporary chunk if it's not the original file
            if chunk_path != pdf_path:
                os.remove(chunk_path)
                
        return png_files
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        raise
    finally:
        # Clean up temporary directory if empty
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)

def rename_and_convert(
    input_folder: str, 
    output_folder: str, 
    valid_pdfs_folder: str, 
    column_name: str = None, 
    valid_types: list = None
):
    """Main workflow function that processes PDFs based on Excel data."""
    try:
        # Create necessary directories
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(valid_pdfs_folder, exist_ok=True)
        temp_dir = os.path.join(valid_pdfs_folder, "temp")
        
        # Get valid PDF files
        pdf_source_folder = os.path.join(input_folder, "PDFAll")
        if not os.path.exists(pdf_source_folder):
            raise FileNotFoundError(f"PDFAll folder not found in {input_folder}")
        
        # Get list of valid files based on Excel data if provided
        valid_files = get_valid_files(
            input_folder, 
            pdf_source_folder, 
            column_name, 
            valid_types
        )
        
        logger.info(f"Found {len(valid_files)} valid PDF files to process")
        
        # Process each PDF
        for pdf_file in valid_files:
            src_path = os.path.join(pdf_source_folder, pdf_file)
            dst_path = os.path.join(valid_pdfs_folder, pdf_file)
            
            # Copy to valid PDFs folder
            shutil.copy2(src_path, dst_path)
            logger.info(f"Copied {pdf_file} to PDFValid folder")
            
            # Process the PDF
            process_pdf(dst_path, output_folder, temp_dir)
            
        logger.info("PDF to PNG conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in rename_and_convert: {str(e)}")
        raise

def get_valid_files(
    input_folder: str, 
    pdf_source_folder: str, 
    column_name: str = None, 
    valid_types: list = None
) -> List[str]:
    """Get list of valid PDF files based on Excel data if provided."""
    if column_name is None or valid_types is None:
        return [f for f in os.listdir(pdf_source_folder) if f.lower().endswith('.pdf')]
    
    excel_files = [f for f in os.listdir(input_folder) if f.endswith('.xlsx')]
    if not excel_files:
        raise FileNotFoundError("No Excel files found in input folder")
    
    excel_path = os.path.join(input_folder, excel_files[0])
    df = pd.read_excel(excel_path)
    
    valid_docs = df[df['FICHERO'].isin(valid_types)]
    valid_pdf_names = [str(name) for name in valid_docs[column_name].dropna().tolist()]
    
    pdf_files = [f for f in os.listdir(pdf_source_folder) if f.lower().endswith('.pdf')]
    return [f for f in pdf_files if any(valid_name in os.path.splitext(f)[0] 
                                      for valid_name in valid_pdf_names)]

def main():
    """CLI entry point that handles argument parsing and executes the conversion workflow."""
    parser = argparse.ArgumentParser(description='Convert PDFs to PNGs with large file support')
    parser.add_argument('-c', '--comunidad', type=str, required=True, help='Name of the Comunidad to process')
    parser.add_argument('-cn', '--column_name', type=str, help='Optional: Name of the column to use for filtering valid PDFs')
    parser.add_argument('-t', '--tipos', type=str, nargs='+', help='Optional: List of valid document types')
    parser.add_argument('-m', '--max_pages', type=int, default=MAX_PAGES_PER_CHUNK, 
                       help=f'Maximum pages per PDF chunk (default: {MAX_PAGES_PER_CHUNK})')
    
    args = parser.parse_args()
    
    # Define paths
    base_path = f"C:/Users/Sergio/GT/AutoAudit/{args.comunidad}"
    paths = {
        'input': base_path,
        'valid_pdfs': os.path.join(base_path, "PDFValid"),
        'output': os.path.join(base_path, "Images")
    }
    
    try:
        rename_and_convert(
            paths['input'],
            paths['output'],
            paths['valid_pdfs'],
            args.column_name,
            args.tipos
        )
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()