import streamlit as st
from PIL import Image
from models.ocr_model import OCRModel
from models.math_solver import MathSolver
from models.explanation_model import ExplanationModel
from utils.image_processing import preprocess_image, segment_image
from utils.pagination_handler import handle_pagination

# Initialize models
ocr_model = OCRModel()
math_solver = MathSolver()
explanation_model = ExplanationModel()

st.title("AI-Powered Multi-Page Math Problem Solver")
st.write("Upload a multi-page image or PDF file containing complex math problems.")

# User Input for Multi-page Problem
uploaded_files = st.file_uploader("Upload one or multiple images", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)

if uploaded_files:
    for page, uploaded_file in enumerate(uploaded_files, start=1):
        st.subheader(f"Processing Page {page}")
        
        # Load and preprocess each page
        image = Image.open(uploaded_file)
        processed_image = preprocess_image(image)
        problem_segments = segment_image(processed_image)
        
        for i, segment in enumerate(problem_segments):
            st.write(f"Detected Problem {i + 1} on Page {page}")
            
            # Extract text
            problem_text = ocr_model.extract_text(segment)
            st.write(f"Problem Text: {problem_text}")
            
            # Solve the problem
            solution = math_solver.solve(problem_text)
            st.write(f"Solution: {solution}")
            
            # Generate explanation
            explanation = explanation_model.generate_explanation(problem_text, solution)
            st.write("Explanation:")
            st.write(explanation)

    st.write("Processing complete for all pages.")
