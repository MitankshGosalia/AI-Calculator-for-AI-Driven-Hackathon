import pytesseract
import torch
import torchvision.transforms as transforms
from PIL import Image

class OCRModel:
    def __init__(self):
        # Load a custom OCR model if available, or configure Tesseract for high accuracy
        self.tesseract_config = '--psm 11'  # Configure for dense text
    
    def extract_text(self, image_segment):
        # Optionally use PyTorch-based OCR if needed; Tesseract for now
        text = pytesseract.image_to_string(image_segment, config=self.tesseract_config)
        return text
