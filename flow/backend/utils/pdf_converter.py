import os
from pathlib import Path
from pdf2image import convert_from_path
import shutil
from typing import List
import logging

logger = logging.getLogger(__name__)

class PDFConverter:
    def __init__(self, upload_dir: str = "./uploads"):
        self.upload_dir = Path(upload_dir)
        self.pdf_dir = self.upload_dir / "PDFAll"
        self.valid_pdf_dir = self.upload_dir / "PDFValid"
        self.images_dir = self.upload_dir / "Images"
        
        # Create directories if they don't exist
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.valid_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    async def convert_pdf_to_images(self, pdf_file: bytes, filename: str) -> List[str]:
        """Convert uploaded PDF to PNG images"""
        try:
            # Save uploaded PDF temporarily
            pdf_path = self.pdf_dir / filename
            with open(pdf_path, 'wb') as f:
                f.write(pdf_file)

            # Convert PDF to images
            images = convert_from_path(str(pdf_path))
            image_paths = []

            # Save each page as PNG
            base_filename = Path(filename).stem
            if len(images) == 1:
                image_path = self.images_dir / f"{base_filename}.png"
                images[0].save(str(image_path), "PNG")
                image_paths.append(image_path.name)
            else:
                for i, image in enumerate(images, 1):
                    image_path = self.images_dir / f"{base_filename}({i}).png"
                    image.save(str(image_path), "PNG")
                    image_paths.append(image_path.name)

            # Copy PDF to valid directory
            shutil.copy2(pdf_path, self.valid_pdf_dir / filename)
            
            return image_paths

        except Exception as e:
            logger.error(f"Error converting PDF {filename}: {str(e)}")
            raise
        finally:
            # Clean up temporary PDF
            if pdf_path.exists():
                pdf_path.unlink() 