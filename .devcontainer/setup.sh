#!/bin/bash
set -e

echo "🚀 Setting up PyCodeRL in GitHub Codespaces..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y gcc binutils build-essential gdb valgrind

# Verify x86-64 architecture
echo "🔍 Verifying architecture..."
echo "Architecture: $(uname -m)"
echo "Platform: $(uname -s)"

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install torch numpy matplotlib seaborn pandas psutil tqdm

# Set up PyCodeRL
echo "⚙️  Setting up PyCodeRL..."
python setup.py

# Create a test script
echo "📝 Creating test script..."
cat > test_pycode_rl.py << 'EOF'
#!/usr/bin/env python3
"""
Quick test of PyCodeRL in Codespaces
"""

from pycode_rl_implementation import PyCodeRLCompiler
import platform

def main():
    print(f"🎯 Testing PyCodeRL on {platform.machine()} {platform.system()}")
    
    test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(f"Fibonacci(10) = {result}")
"""
    
    compiler = PyCodeRLCompiler()
    
    try:
        print("🔥 Compiling Python to x86 assembly...")
        assembly, metrics = compiler.compile_to_machine_code(test_code)
        
        print(f"✅ Compilation successful!")
        print(f"   - Compilation time: {metrics['compilation_time']:.3f}s")
        print(f"   - Instructions generated: {metrics['instruction_count']}")
        print(f"   - Assembly size: {metrics['code_size']} chars")
        
        print("🏃 Testing execution...")
        exec_metrics = compiler.execute_and_evaluate(assembly)
        
        if exec_metrics['correctness_score'] > 0:
            print(f"✅ Execution successful!")
            print(f"   - Execution time: {exec_metrics['execution_time']:.3f}s")
            print(f"   - Output: {exec_metrics['output'].strip()}")
        else:
            print(f"❌ Execution failed")
            print(f"   - Errors: {exec_metrics['errors']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
EOF

chmod +x test_pycode_rl.py

echo "✅ PyCodeRL setup complete!"
echo ""
echo "🎉 Next steps:"
echo "   1. Run: python test_pycode_rl.py"
echo "   2. Try: python examples/example_simple.py"
echo "   3. Start training: python examples/example_training.py"
echo ""
echo "📚 Files created:"
echo "   - test_pycode_rl.py (quick test script)"
echo "   - examples/ (example scripts)"
echo "   - models/ (for saved models)"
echo ""
