import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import pytesseract
import numpy as np
import cv2

# Set Streamlit configuration
st.set_page_config(
    page_title="AI-Powered Math Problem Solver",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define a simple CNN model for character recognition using PyTorch
class CharacterRecognitionModel(nn.Module):
    def __init__(self):
        super(CharacterRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 36)  # 36 classes: 10 digits + 26 uppercase letters

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model and load pre-trained weights (dummy for demonstration)
model = CharacterRecognitionModel()
model.load_state_dict(torch.load("character_model.pth"))  # Assume model weights are saved here
model.eval()

# Define image preprocessing for PyTorch model
def preprocess_image_for_model(image):
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return preprocess(image)

# OCR function using PyTorch character recognition
def pytorch_ocr(image):
    image = np.array(image.convert('L'))  # Convert to grayscale
    _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    recognized_text = ""

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:  # Filter small contours
            char_image = Image.fromarray(thresh[y:y+h, x:x+w])
            input_tensor = preprocess_image_for_model(char_image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                recognized_text += chr(predicted.item() + 48)  # Convert to character

    return recognized_text

# Streamlit Sidebar
st.sidebar.title("AI-Powered Math Problem Solver 🧠📐")
st.sidebar.write("Upload a multi-page image or PDF with math problems and get detailed solutions.")
st.sidebar.image("https://via.placeholder.com/250x250.png?text=Math+Solver+Logo", use_column_width=True)

# Main Header
st.title("Welcome to the AI-Powered Math Problem Solver")
st.markdown("""
This app allows you to upload images or PDFs containing complex math problems and provides step-by-step solutions.
Get instant explanations for calculus, algebra, differential equations, and more!
""")

# File Uploader
uploaded_files = st.file_uploader("Upload your math problem images or PDFs here:", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)
if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} file(s). Processing...")

    # Process each uploaded file
    for page_num, uploaded_file in enumerate(uploaded_files, start=1):
        st.markdown(f"### Page {page_num}")
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Page {page_num}", use_column_width=True)

        # Perform OCR using PyTorch model
        problem_text = pytorch_ocr(image)
        
        # Display detected text (for now, only display recognized characters)
        st.write(f"Detected Problem Text: `{problem_text}`")

        # Example solution and explanation (placeholders)
        st.subheader("Solution")
        st.write("Solution goes here (integrate or solve using SymPy).")

        st.subheader("Explanation")
        st.write("""
        - Step-by-Step Explanation:
            - **Step 1**: Recognize characters in the equation.
            - **Step 2**: Parse the equation and perform symbolic computation.
            - **Final Solution**: Present the solution with each step explained.
        """)

    st.success("Processing complete for all pages!")

# Footer
st.markdown("---")
st.write("**AI-Powered Math Problem Solver** © 2023 | Powered by PyTorch, OCR, and Symbolic Computation")

# CSS for custom layout and styling
st.markdown("""
<style>
    .css-1d391kg {
        font-family: 'Arial', sans-serif;
    }
    .css-12w0qpk {
        color: #4A90E2;
    }
    .css-1v3fvcr {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 20px;
    }
    .css-1b5izki {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)
