import cv2
import numpy as np
from PIL import Image

def preprocess_image(image):
    # Convert to grayscale and threshold
    image = np.array(image.convert('L'))
    _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    return Image.fromarray(thresh)

def segment_image(processed_image):
    # Implement image segmentation for pages with multiple math problems
    contours, _ = cv2.findContours(np.array(processed_image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        segment = processed_image.crop((x, y, x + w, y + h))
        segments.append(segment)
    return segments
