import pytesseract
import cv2
import numpy as np
from PIL import Image

class TextExtractor:
    def __init__(self):
        # Configuration for Tesseract OCR
        self.config = '--psm 6'  # Page segmentation mode for single blocks of text
    
    def preprocess_image_for_ocr(self, image):
        """
        Preprocesses the image to improve OCR accuracy:
        1. Convert to grayscale.
        2. Apply Gaussian Blur.
        3. Thresholding for binary image.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return thresh
    
    def extract_text(self, image):
        """
        Extracts text from an image using Tesseract OCR after preprocessing.
        """
        # Preprocess the image for OCR
        preprocessed_image = self.preprocess_image_for_ocr(image)
        
        # Use Tesseract to extract text
        text = pytesseract.image_to_string(preprocessed_image, config=self.config)
        
        # Optional: Clean up text output by removing unnecessary characters
        cleaned_text = self.clean_text(text)
        
        return cleaned_text
    
    def clean_text(self, text):
        """
        Cleans the extracted text, removing unwanted characters and symbols
        that may interfere with math parsing.
        """
        # Remove non-standard characters often misinterpreted by OCR
        cleaned_text = ''.join(c for c in text if c.isalnum() or c in "+-*/=()[]{}., ")
        
        # Replace multiple spaces with a single space
        cleaned_text = ' '.join(cleaned_text.split())
        
        return cleaned_text
