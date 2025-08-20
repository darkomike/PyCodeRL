#!/usr/bin/env python3
"""
PyCodeRL Setup and Installation Script
Sets up the environment and dependencies for PyCodeRL
"""

import subprocess
import sys
import os
import platform
from pathlib import Path
import argparse
import tempfile

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("ERROR: PyCodeRL requires Python 3.8 or higher")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    return True

def check_system_dependencies():
    """Check for required system dependencies"""
    dependencies = {
        'gcc': ('GCC compiler for assembly compilation', '--version'),
        'as': ('GNU assembler (part of binutils)', '--version'),
        'ld': ('GNU linker (part of binutils)', '-v' if platform.system() == 'Darwin' else '--version')
    }
    
    missing = []
    for cmd, (description, version_flag) in dependencies.items():
        try:
            subprocess.run([cmd, version_flag], capture_output=True, check=True)
            print(f"✓ {cmd} found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"✗ {cmd} not found ({description})")
            missing.append(cmd)
    
    return missing

def install_python_dependencies():
    """Install required Python packages"""
    dependencies = [
        'torch>=1.9.0',
        'numpy>=1.20.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'pandas>=1.3.0',
        'psutil>=5.8.0',
        'tqdm>=4.60.0'
    ]
    
    print("Installing Python dependencies...")
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', dep
            ], check=True, capture_output=True)
            print(f"✓ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {dep}: {e}")
            return False
    
    return True

def setup_development_environment():
    """Set up development environment"""
    print("Setting up development environment...")
    
    # Create necessary directories
    directories = [
        'models',
        'checkpoints', 
        'datasets',
        'results',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Create example configuration file
    config_content = """
# PyCodeRL Configuration
{
    "training": {
        "learning_rate": 3e-4,
        "batch_size": 32,
        "num_episodes": 2000,
        "eval_interval": 100,
        "clip_epsilon": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95
    },
    "model": {
        "state_dim": 512,
        "hidden_dim": 256,
        "num_registers": 12,
        "instruction_types": 16
    },
    "compilation": {
        "optimization_level": "balanced",
        "target_architecture": "x86_64",
        "enable_simd": true,
        "max_compilation_time": 30.0
    },
    "evaluation": {
        "benchmark_iterations": 5,
        "timeout_seconds": 30,
        "enable_profiling": true
    }
}
"""
    
    with open('config.json', 'w') as f:
        f.write(config_content.strip())
    print("✓ Created config.json")
    
    return True

def test_installation():
    """Test if PyCodeRL is working correctly"""
    print("Testing PyCodeRL installation...")
    
    # Check architecture - now we support ARM64!
    if platform.machine() == 'arm64' and platform.system() == 'Darwin':
        print("✅ Running on ARM64 Mac - PyCodeRL is now natively supported!")
    else:
        print(f"ℹ️  Running on {platform.machine()} {platform.system()}")
    
    # Simple test program
    test_code = """
def test_function(x, y):
    result = x + y
    return result

answer = test_function(5, 3)
"""
    
    try:
        # Import and test basic functionality
        sys.path.append(os.getcwd())
        from pycode_rl_implementation import PyCodeRLCompiler
        
        compiler = PyCodeRLCompiler()
        
        # Test compilation
        assembly, metrics = compiler.compile_to_machine_code(test_code)
        
        if assembly and metrics['compilation_time'] > 0:
            print("✓ Basic compilation test passed")
            
            # Test execution on ARM64
            exec_metrics = compiler.execute_and_evaluate(assembly)
            if exec_metrics['correctness_score'] > 0:
                print("✓ Basic execution test passed")
                print("🎉 PyCodeRL ARM64 implementation is working!")
                return True
            else:
                print("✗ Execution test failed")
                print(f"   Errors: {exec_metrics.get('errors', 'Unknown error')}")
                return False
        else:
            print("✗ Compilation test failed")
            return False
            
    except Exception as e:
        print(f"✗ Installation test failed: {e}")
        return False

def install_optional_dependencies():
    """Install optional dependencies for enhanced functionality"""
    optional_deps = {
        'pypy3': 'PyPy for performance comparison',
        'numba': 'Numba for numerical Python optimization',
        'cython': 'Cython for Python-to-C compilation'
    }
    
    print("Installing optional dependencies...")
    
    # Try to install PyPy
    system = platform.system().lower()
    if system == 'linux':
        try:
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'pypy3'], 
                         check=True, capture_output=True)
            print("✓ PyPy3 installed")
        except subprocess.CalledProcessError:
            print("✗ Could not install PyPy3 (try manual installation)")
    elif system == 'darwin':  # macOS
        try:
            subprocess.run(['brew', 'install', 'pypy3'], 
                         check=True, capture_output=True)
            print("✓ PyPy3 installed")
        except subprocess.CalledProcessError:
            print("✗ Could not install PyPy3 (try: brew install pypy3)")
    else:
        print("ℹ PyPy3 installation not automated for this system")
    
    # Install Python packages
    python_optional = ['numba', 'cython']
    for package in python_optional:
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', package
            ], check=True, capture_output=True)
            print(f"✓ {package} installed")
        except subprocess.CalledProcessError:
            print(f"✗ Could not install {package}")

def create_example_scripts():
    """Create example usage scripts"""
    
    # Simple usage example
    simple_example = '''#!/usr/bin/env python3
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
'''
    
    # Training example
    training_example = '''#!/usr/bin/env python3
"""
PyCodeRL Training Example
"""

from pycode_rl_training import PPOTrainer, PythonProgramGenerator
from pycode_rl_implementation import PyCodeRLCompiler

def main():
    # Initialize compiler and trainer
    compiler = PyCodeRLCompiler()
    trainer = PPOTrainer(compiler)
    
    # Generate training programs
    generator = PythonProgramGenerator()
    training_programs = generator.generate_dataset(100)
    
    print(f"Generated {len(training_programs)} training programs")
    
    # Start training
    trainer.train(
        training_programs=training_programs,
        num_episodes=500,
        batch_size=16,
        eval_interval=50
    )
    
    # Save trained model
    trainer.save_checkpoint("trained_model.pt")
    
    # Plot training progress
    trainer.plot_training_metrics()

if __name__ == "__main__":
    main()
'''
    
    # Evaluation example
    eval_example = '''#!/usr/bin/env python3
"""
PyCodeRL Evaluation Example
"""

from pycode_rl_evaluation import BenchmarkSuite, PythonImplementationRunner, PerformanceAnalyzer

def main():
    # Initialize components
    suite = BenchmarkSuite()
    runner = PythonImplementationRunner()
    analyzer = PerformanceAnalyzer()
    
    # Select benchmarks to run
    benchmarks_to_run = ['fibonacci', 'factorial', 'list_operations']
    implementations = ['cpython', 'pycode_rl']
    
    print("Running performance evaluation...")
    
    # Run benchmarks
    all_results = []
    for benchmark_name in benchmarks_to_run:
        if benchmark_name in suite.benchmarks:
            code = suite.benchmarks[benchmark_name]
            results = runner.run_benchmark(benchmark_name, code, implementations)
            all_results.extend(results)
    
    # Analyze results
    analyzer.add_results(all_results)
    report = analyzer.generate_report()
    
    # Display summary
    print("\\nPerformance Summary:")
    for impl, stats in report.get('summary', {}).items():
        print(f"{impl}: {stats['mean_speedup']:.2f}x average speedup")
    
    # Create visualizations
    analyzer.plot_performance_comparison()

if __name__ == "__main__":
    main()
'''
    
    # Write example scripts
    examples = {
        'example_simple.py': simple_example,
        'example_training.py': training_example,
        'example_evaluation.py': eval_example
    }
    
    examples_dir = Path('examples')
    examples_dir.mkdir(exist_ok=True)
    
    for filename, content in examples.items():
        filepath = examples_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
        filepath.chmod(0o755)  # Make executable
        print(f"✓ Created {filepath}")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='PyCodeRL Setup and Installation')
    parser.add_argument('--skip-deps', action='store_true', 
                       help='Skip dependency installation')
    parser.add_argument('--skip-test', action='store_true',
                       help='Skip installation test')
    parser.add_argument('--install-optional', action='store_true',
                       help='Install optional dependencies')
    
    args = parser.parse_args()
    
    print("PyCodeRL Setup and Installation")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check system dependencies
    print("\\nChecking system dependencies...")
    missing_deps = check_system_dependencies()
    
    if missing_deps:
        print(f"\\nMissing dependencies: {', '.join(missing_deps)}")
        print("Please install them using your system package manager:")
        
        system = platform.system().lower()
        if system == 'linux':
            print("  Ubuntu/Debian: sudo apt-get install gcc binutils")
            print("  CentOS/RHEL: sudo yum install gcc binutils")
        elif system == 'darwin':
            print("  macOS: xcode-select --install")
        else:
            print("  Windows: Install MinGW-w64 or Visual Studio Build Tools")
        
        return 1
    
    # Install Python dependencies
    if not args.skip_deps:
        print("\\nInstalling Python dependencies...")
        if not install_python_dependencies():
            print("Dependency installation failed")
            return 1
    
    # Install optional dependencies
    if args.install_optional:
        print("\\nInstalling optional dependencies...")
        install_optional_dependencies()
    
    # Setup development environment
    print("\\nSetting up development environment...")
    if not setup_development_environment():
        print("Environment setup failed")
        return 1
    
    # Create example scripts
    print("\\nCreating example scripts...")
    create_example_scripts()
    
    # Test installation
    if not args.skip_test:
        print("\\nTesting installation...")
        if not test_installation():
            print("Installation test failed")
            return 1
    
    print("\\n" + "=" * 40)
    print("PyCodeRL setup completed successfully!")
    print("\\nQuick start:")
    print("1. Run a simple example: python examples/example_simple.py")
    print("2. Start training: python examples/example_training.py")
    print("3. Run evaluation: python examples/example_evaluation.py")
    print("\\nFor more information, see the documentation.")
    
    return 0

if __name__ == "__main__":
    exit(main())