from transformers import pipeline

class ExplanationModel:
    def __init__(self):
        self.generator = pipeline('text-generation', model='gpt-2')  # Replace with fine-tuned model if available
    
    def generate_explanation(self, problem_text, solution):
        prompt = f"Provide a detailed solution for: {problem_text}. Solution is: {solution}."
        explanation = self.generator(prompt, max_length=200)[0]['generated_text']
        return explanation
