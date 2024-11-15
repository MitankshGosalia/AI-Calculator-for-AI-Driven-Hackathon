from sympy import sympify, simplify, solve, Integral, symbols
from sympy.solvers import solve
import sympy

class MathSolver:
    def __init__(self):
        pass
    
    def solve(self, problem_text):
        try:
            # Parse the text to a SymPy-compatible format
            expression = sympify(problem_text)
            
            if isinstance(expression, sympy.Equality):
                solution = solve(expression)
            elif isinstance(expression, Integral):
                solution = expression.doit()  # Solve integrals
            else:
                solution = simplify(expression)  # Simplify other expressions
            
            return solution
        except Exception as e:
            return f"Error parsing or solving problem: {e}"
