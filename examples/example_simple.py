#!/usr/bin/env python3
"""
Simple PyCodeRL Usage Example
"""

from pycode_rl_implementation import PyCodeRLCompiler

def main():
    # Initialize compiler
    compiler = PyCodeRLCompiler()
    
    # Example Python code
    python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(f"Fibonacci(10) = {result}")
"""
    
    print("Compiling Python code with PyCodeRL...")
    
    # Compile to assembly
    assembly, metrics = compiler.compile_to_machine_code(python_code)
    
    print(f"Compilation completed in {metrics['compilation_time']:.4f} seconds")
    print(f"Generated {metrics['instruction_count']} instructions")
    
    # Execute compiled code
    exec_metrics = compiler.execute_and_evaluate(assembly)
    
    print(f"Execution time: {exec_metrics['execution_time']:.4f} seconds")
    print(f"Correctness score: {exec_metrics['correctness_score']}")
    print(f"Output: {exec_metrics['output']}")

if __name__ == "__main__":
    main()
