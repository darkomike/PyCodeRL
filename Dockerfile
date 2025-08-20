# PyCodeRL Docker Image for x86-64 execution
FROM --platform=linux/amd64 python:3.11-slim-bullseye

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PYCODE_RL_ENV=docker

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    binutils \
    build-essential \
    gdb \
    valgrind \
    git \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p models checkpoints datasets results logs

# Create a comprehensive test script
RUN cat > docker_test.py << 'EOF'
#!/usr/bin/env python3
"""
Comprehensive Docker test for PyCodeRL
"""

from pycode_rl_implementation import PyCodeRLCompiler
import platform
import sys
import time

def test_basic_compilation():
    """Test basic compilation functionality"""
    print("🧪 Testing basic compilation...")
    
    test_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
print(f"5! = {result}")
"""
    
    compiler = PyCodeRLCompiler()
    assembly, metrics = compiler.compile_to_machine_code(test_code)
    
    print(f"   ✅ Compiled {metrics['instruction_count']} instructions")
    print(f"   ⏱️  Compilation time: {metrics['compilation_time']:.3f}s")
    return assembly

def test_execution(assembly):
    """Test assembly execution"""
    print("🏃 Testing execution...")
    
    compiler = PyCodeRLCompiler()
    exec_metrics = compiler.execute_and_evaluate(assembly)
    
    if exec_metrics['correctness_score'] > 0:
        print(f"   ✅ Execution successful!")
        print(f"   ⏱️  Execution time: {exec_metrics['execution_time']:.3f}s")
        if exec_metrics.get('output'):
            print(f"   📄 Output: {exec_metrics['output'].strip()}")
        return True
    else:
        print(f"   ❌ Execution failed")
        if exec_metrics.get('errors'):
            print(f"   🐛 Errors: {exec_metrics['errors']}")
        return False

def main():
    print("🐳 PyCodeRL Docker Test Suite")
    print("=" * 50)
    print(f"Platform: {platform.machine()} {platform.system()}")
    print(f"Python: {sys.version}")
    print(f"Architecture: {platform.architecture()}")
    print()
    
    try:
        # Test compilation
        assembly = test_basic_compilation()
        
        # Test execution  
        execution_success = test_execution(assembly)
        
        if execution_success:
            print("\n🎉 All tests passed! PyCodeRL is working correctly in Docker.")
            return True
        else:
            print("\n⚠️  Compilation works, but execution failed.")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print("\n" + "=" * 50)
    if success:
        print("✅ PyCodeRL Docker container is ready!")
    else:
        print("❌ PyCodeRL Docker container has issues.")
    sys.exit(0 if success else 1)
EOF

# Make scripts executable
RUN chmod +x docker_test.py

# Run a basic setup check (without full execution test)
RUN python -c "
import sys, os
sys.path.append(os.getcwd())
from pycode_rl_implementation import PyCodeRLCompiler
compiler = PyCodeRLCompiler()
print('✅ PyCodeRL imports successfully in Docker')
print(f'🏗️  Architecture: {os.uname().machine}')
"

# Expose ports for potential web interfaces
EXPOSE 8000 8080 8888

# Set the default command to run comprehensive tests
CMD ["python", "docker_test.py"]

# Add metadata labels
LABEL org.opencontainers.image.title="PyCodeRL"
LABEL org.opencontainers.image.description="Direct Python to x86 Machine Code Generator using Reinforcement Learning"
LABEL org.opencontainers.image.version="1.0"
LABEL org.opencontainers.image.source="https://github.com/darkomike/PyCodeRL"
