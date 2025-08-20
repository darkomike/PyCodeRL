#!/usr/bin/env python3
"""
PyCodeRL Evaluation and Benchmarking Suite
Comprehensive evaluation against CPython, PyPy, and other Python implementations
"""

import time
import subprocess
import tempfile
import os
import statistics
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataclasses import dataclass
import psutil
import traceback

from pycode_rl_implementation import PyCodeRLCompiler

@dataclass
class BenchmarkResult:
    """Results from running a benchmark"""
    name: str
    implementation: str
    execution_time: float
    compilation_time: float
    memory_usage: float
    correctness_score: float
    output: str
    error: Optional[str] = None

class PythonImplementationRunner:
    """Runs Python code on different implementations"""
    
    def __init__(self):
        self.implementations = {
            'cpython': self._run_cpython,
            'pypy': self._run_pypy,
            'pycode_rl': self._run_pycode_rl
        }
        
        # Initialize PyCodeRL compiler
        self.pycode_rl_compiler = PyCodeRLCompiler()
        
    def _run_cpython(self, code: str, iterations: int = 1) -> BenchmarkResult:
        """Run code with CPython"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Wrap code with timing
                timed_code = f"""
import time
import psutil
import os

process = psutil.Process(os.getpid())
start_memory = process.memory_info().rss

start_time = time.time()
for _ in range({iterations}):
{self._indent_code(code, 4)}
end_time = time.time()

end_memory = process.memory_info().rss
print(f"TIMING_INFO:{{end_time - start_time:.6f}}")
print(f"MEMORY_INFO:{{end_memory - start_memory}}")
"""
                f.write(timed_code)
                f.flush()
                
                # Run with CPython
                start_time = time.time()
                result = subprocess.run([
                    sys.executable, f.name
                ], capture_output=True, text=True, timeout=30)
                compilation_time = time.time() - start_time
                
                # Parse timing info
                execution_time, memory_usage = self._parse_timing_info(result.stdout)
                
                os.unlink(f.name)
                
                return BenchmarkResult(
                    name="test",
                    implementation="cpython",
                    execution_time=execution_time,
                    compilation_time=compilation_time,
                    memory_usage=memory_usage,
                    correctness_score=1.0 if result.returncode == 0 else 0.0,
                    output=result.stdout,
                    error=result.stderr if result.stderr else None
                )
                
        except Exception as e:
            return BenchmarkResult(
                name="test", implementation="cpython",
                execution_time=float('inf'), compilation_time=float('inf'),
                memory_usage=0, correctness_score=0.0,
                output="", error=str(e)
            )
    
    def _run_pypy(self, code: str, iterations: int = 1) -> BenchmarkResult:
        """Run code with PyPy (if available)"""
        try:
            # Check if PyPy is available
            subprocess.run(['pypy3', '--version'], capture_output=True, check=True)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                timed_code = f"""
import time
import gc

start_time = time.time()
for _ in range({iterations}):
{self._indent_code(code, 4)}
end_time = time.time()

print(f"TIMING_INFO:{{end_time - start_time:.6f}}")
print(f"MEMORY_INFO:0")  # Simplified for PyPy
"""
                f.write(timed_code)
                f.flush()
                
                start_time = time.time()
                result = subprocess.run([
                    'pypy3', f.name
                ], capture_output=True, text=True, timeout=30)
                compilation_time = time.time() - start_time
                
                execution_time, memory_usage = self._parse_timing_info(result.stdout)
                
                os.unlink(f.name)
                
                return BenchmarkResult(
                    name="test",
                    implementation="pypy",
                    execution_time=execution_time,
                    compilation_time=compilation_time,
                    memory_usage=memory_usage,
                    correctness_score=1.0 if result.returncode == 0 else 0.0,
                    output=result.stdout,
                    error=result.stderr if result.stderr else None
                )
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            # PyPy not available
            return BenchmarkResult(
                name="test", implementation="pypy",
                execution_time=float('inf'), compilation_time=float('inf'),
                memory_usage=0, correctness_score=0.0,
                output="", error="PyPy not available"
            )
        except Exception as e:
            return BenchmarkResult(
                name="test", implementation="pypy",
                execution_time=float('inf'), compilation_time=float('inf'),
                memory_usage=0, correctness_score=0.0,
                output="", error=str(e)
            )
    
    def _run_pycode_rl(self, code: str, iterations: int = 1) -> BenchmarkResult:
        """Run code with PyCodeRL"""
        try:
            start_time = time.time()
            
            # Compile with PyCodeRL
            assembly, compile_metrics = self.pycode_rl_compiler.compile_to_machine_code(code)
            compilation_time = time.time() - start_time
            
            # Execute compiled code multiple times
            total_execution_time = 0
            total_memory_usage = 0
            output = ""
            
            for _ in range(iterations):
                exec_metrics = self.pycode_rl_compiler.execute_and_evaluate(assembly)
                total_execution_time += exec_metrics['execution_time']
                output = exec_metrics['output']
                
            return BenchmarkResult(
                name="test",
                implementation="pycode_rl",
                execution_time=total_execution_time,
                compilation_time=compilation_time,
                memory_usage=total_memory_usage,
                correctness_score=exec_metrics['correctness_score'],
                output=output,
                error=exec_metrics.get('errors')
            )
            
        except Exception as e:
            return BenchmarkResult(
                name="test", implementation="pycode_rl",
                execution_time=float('inf'), compilation_time=float('inf'),
                memory_usage=0, correctness_score=0.0,
                output="", error=str(e)
            )
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces"""
        lines = code.strip().split('\n')
        return '\n'.join(' ' * spaces + line for line in lines)
    
    def _parse_timing_info(self, output: str) -> Tuple[float, float]:
        """Parse timing information from output"""
        execution_time = 0.0
        memory_usage = 0.0
        
        for line in output.split('\n'):
            if line.startswith('TIMING_INFO:'):
                execution_time = float(line.split(':')[1])
            elif line.startswith('MEMORY_INFO:'):
                memory_usage = float(line.split(':')[1])
                
        return execution_time, memory_usage
    
    def run_benchmark(self, name: str, code: str, implementations: List[str], 
                     iterations: int = 1) -> List[BenchmarkResult]:
        """Run benchmark on specified implementations"""
        results = []
        
        for impl in implementations:
            if impl in self.implementations:
                print(f"  Running on {impl}...")
                result = self.implementations[impl](code, iterations)
                result.name = name
                results.append(result)
            else:
                print(f"  Implementation {impl} not available")
                
        return results

class BenchmarkSuite:
    """Collection of Python benchmarks for evaluation"""
    
    def __init__(self):
        self.benchmarks = {
            # Arithmetic benchmarks
            'fibonacci': """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(20)
""",
            
            'factorial': """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

result = factorial(10)
""",
            
            'prime_sieve': """
def sieve_of_eratosthenes(limit):
    primes = [True] * (limit + 1)
    primes[0] = primes[1] = False
    
    for i in range(2, int(limit**0.5) + 1):
        if primes[i]:
            for j in range(i*i, limit + 1, i):
                primes[j] = False
    
    return [i for i in range(2, limit + 1) if primes[i]]

result = sieve_of_eratosthenes(1000)
""",
            
            # List operations
            'list_operations': """
numbers = list(range(1000))
result = sum(x * x for x in numbers if x % 2 == 0)
""",
            
            'list_comprehension': """
data = list(range(100))
result = [x*2 + 1 for x in data if x % 3 == 0]
""",
            
            # String operations
            'string_processing': """
text = "hello world " * 100
words = text.split()
result = " ".join(word.upper() for word in words)
""",
            
            # Mathematical operations
            'matrix_multiply': """
def matrix_multiply(a, b):
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    
    return result

a = [[1, 2], [3, 4]]
b = [[5, 6], [7, 8]]
result = matrix_multiply(a, b)
""",
            
            # Sorting algorithms
            'quicksort': """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

data = [3, 6, 8, 10, 1, 2, 1]
result = quicksort(data)
""",
            
            # Object-oriented programming
            'class_operations': """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5

points = [Point(i, i*2) for i in range(50)]
total_distance = sum(points[i].distance(points[i+1]) 
                    for i in range(len(points)-1))
""",
            
            # Control flow heavy
            'nested_loops': """
total = 0
for i in range(20):
    for j in range(20):
        for k in range(10):
            total += i * j * k

result = total
"""
        }

class PerformanceAnalyzer:
    """Analyzes and visualizes benchmark results"""
    
    def __init__(self):
        self.results = []
        
    def add_results(self, results: List[BenchmarkResult]):
        """Add benchmark results"""
        self.results.extend(results)
    
    def generate_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if not self.results:
            return {}
        
        # Group results by benchmark and implementation
        grouped_results = {}
        for result in self.results:
            if result.name not in grouped_results:
                grouped_results[result.name] = {}
            grouped_results[result.name][result.implementation] = result
        
        # Calculate speedups relative to CPython
        report = {
            'benchmarks': {},
            'summary': {},
            'analysis': {}
        }
        
        total_speedups = {'pypy': [], 'pycode_rl': []}
        
        for benchmark_name, impl_results in grouped_results.items():
            if 'cpython' not in impl_results:
                continue
                
            cpython_time = impl_results['cpython'].execution_time
            
            benchmark_data = {
                'cpython': {
                    'execution_time': cpython_time,
                    'compilation_time': impl_results['cpython'].compilation_time,
                    'correctness': impl_results['cpython'].correctness_score
                }
            }
            
            for impl in ['pypy', 'pycode_rl']:
                if impl in impl_results and impl_results[impl].execution_time > 0:
                    speedup = cpython_time / impl_results[impl].execution_time
                    total_speedups[impl].append(speedup)
                    
                    benchmark_data[impl] = {
                        'execution_time': impl_results[impl].execution_time,
                        'compilation_time': impl_results[impl].compilation_time,
                        'correctness': impl_results[impl].correctness_score,
                        'speedup': speedup
                    }
            
            report['benchmarks'][benchmark_name] = benchmark_data
        
        # Calculate summary statistics
        for impl in ['pypy', 'pycode_rl']:
            if total_speedups[impl]:
                report['summary'][impl] = {
                    'mean_speedup': statistics.mean(total_speedups[impl]),
                    'median_speedup': statistics.median(total_speedups[impl]),
                    'min_speedup': min(total_speedups[impl]),
                    'max_speedup': max(total_speedups[impl]),
                    'std_speedup': statistics.stdev(total_speedups[impl]) if len(total_speedups[impl]) > 1 else 0
                }
        
        return report
    
    def plot_performance_comparison(self, save_path: str = "performance_comparison.png"):
        """Create performance comparison visualizations"""
        if not self.results:
            print("No results to plot")
            return
        
        # Prepare data for plotting
        df_data = []
        for result in self.results:
            if result.execution_time < float('inf') and result.correctness_score > 0:
                df_data.append({
                    'benchmark': result.name,
                    'implementation': result.implementation,
                    'execution_time': result.execution_time,
                    'compilation_time': result.compilation_time,
                    'correctness': result.correctness_score
                })
        
        if not df_data:
            print("No valid results to plot")
            return
        
        df = pd.DataFrame(df_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Execution time comparison
        sns.barplot(data=df, x='benchmark', y='execution_time', hue='implementation', ax=axes[0, 0])
        axes[0, 0].set_title('Execution Time Comparison')
        axes[0, 0].set_ylabel('Execution Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Compilation time comparison
        sns.barplot(data=df, x='benchmark', y='compilation_time', hue='implementation', ax=axes[0, 1])
        axes[0, 1].set_title('Compilation Time Comparison')
        axes[0, 1].set_ylabel('Compilation Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Speedup calculation and visualization
        speedup_data = []
        for benchmark in df['benchmark'].unique():
            benchmark_df = df[df['benchmark'] == benchmark]
            cpython_time = benchmark_df[benchmark_df['implementation'] == 'cpython']['execution_time']
            
            if not cpython_time.empty:
                baseline = cpython_time.iloc[0]
                for _, row in benchmark_df.iterrows():
                    if row['implementation'] != 'cpython':
                        speedup = baseline / row['execution_time'] if row['execution_time'] > 0 else 0
                        speedup_data.append({
                            'benchmark': benchmark,
                            'implementation': row['implementation'],
                            'speedup': speedup
                        })
        
        if speedup_data:
            speedup_df = pd.DataFrame(speedup_data)
            sns.barplot(data=speedup_df, x='benchmark', y='speedup', hue='implementation', ax=axes[1, 0])
            axes[1, 0].set_title('Speedup over CPython')
            axes[1, 0].set_ylabel('Speedup Factor')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='CPython baseline')
        
        # Correctness scores
        sns.barplot(data=df, x='benchmark', y='correctness', hue='implementation', ax=axes[1, 1])
        axes[1, 1].set_title('Correctness Scores')
        axes[1, 1].set_ylabel('Correctness Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Performance comparison plot saved to {save_path}")
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save results to JSON file"""
        results_data = []
        for result in self.results:
            results_data.append({
                'name': result.name,
                'implementation': result.implementation,
                'execution_time': result.execution_time,
                'compilation_time': result.compilation_time,
                'memory_usage': result.memory_usage,
                'correctness_score': result.correctness_score,
                'output': result.output,
                'error': result.error
            })
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to {filename}")

def main():
    """Main evaluation script"""
    
    print("PyCodeRL Evaluation and Benchmarking Suite")
    print("=" * 50)
    
    # Initialize components
    runner = PythonImplementationRunner()
    suite = BenchmarkSuite()
    analyzer = PerformanceAnalyzer()
    
    # Available implementations
    implementations = ['cpython', 'pypy', 'pycode_rl']
    
    print(f"Available implementations: {implementations}")
    print(f"Available benchmarks: {list(suite.benchmarks.keys())}")
    print()
    
    # Run benchmarks
    all_results = []
    
    for benchmark_name, code in suite.benchmarks.items():
        print(f"Running benchmark: {benchmark_name}")
        print("-" * 30)
        
        try:
            results = runner.run_benchmark(
                benchmark_name, 
                code, 
                implementations,
                iterations=3  # Run each benchmark 3 times for averaging
            )
            
            all_results.extend(results)
            
            # Display immediate results
            for result in results:
                status = "✓" if result.correctness_score > 0.5 else "✗"
                print(f"  {result.implementation:12} {status} "
                      f"Exec: {result.execution_time:.4f}s "
                      f"Compile: {result.compilation_time:.4f}s "
                      f"Correct: {result.correctness_score:.2f}")
                
                if result.error:
                    print(f"    Error: {result.error}")
            
            print()
            
        except Exception as e:
            print(f"  Benchmark failed: {e}")
            traceback.print_exc()
            print()
    
    # Add results to analyzer
    analyzer.add_results(all_results)
    
    # Generate comprehensive report
    print("Generating Performance Report...")
    print("=" * 40)
    
    report = analyzer.generate_report()
    
    if report:
        # Print summary
        print("Summary Statistics:")
        print("-" * 20)
        
        for impl, stats in report.get('summary', {}).items():
            print(f"{impl.upper()}:")
            print(f"  Mean speedup over CPython: {stats['mean_speedup']:.2f}×")
            print(f"  Median speedup: {stats['median_speedup']:.2f}×")
            print(f"  Range: {stats['min_speedup']:.2f}× - {stats['max_speedup']:.2f}×")
            print(f"  Standard deviation: {stats['std_speedup']:.2f}")
            print()
        
        # Print detailed benchmark results
        print("Detailed Results:")
        print("-" * 20)
        
        for benchmark_name, bench_data in report.get('benchmarks', {}).items():
            print(f"{benchmark_name}:")
            
            cpython_time = bench_data.get('cpython', {}).get('execution_time', 0)
            print(f"  CPython: {cpython_time:.4f}s")
            
            for impl in ['pypy', 'pycode_rl']:
                if impl in bench_data:
                    data = bench_data[impl]
                    print(f"  {impl}: {data['execution_time']:.4f}s "
                          f"(speedup: {data.get('speedup', 0):.2f}×)")
            print()
    
    # Create visualizations
    print("Creating visualizations...")
    analyzer.plot_performance_comparison("pycode_rl_performance.png")
    
    # Save results
    analyzer.save_results("pycode_rl_benchmark_results.json")
    
    # Save detailed report
    with open("pycode_rl_performance_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print("Evaluation completed!")
    print("\nFiles generated:")
    print("- pycode_rl_performance.png (performance charts)")
    print("- pycode_rl_benchmark_results.json (raw results)")
    print("- pycode_rl_performance_report.json (detailed analysis)")
    
    # Performance insights
    if report.get('summary'):
        print("\nKey Insights:")
        print("-" * 15)
        
        pycode_rl_stats = report['summary'].get('pycode_rl')
        pypy_stats = report['summary'].get('pypy')
        
        if pycode_rl_stats:
            mean_speedup = pycode_rl_stats['mean_speedup']
            if mean_speedup > 1.0:
                print(f"• PyCodeRL achieved {mean_speedup:.2f}× average speedup over CPython")
            else:
                print(f"• PyCodeRL showed {1/mean_speedup:.2f}× slowdown compared to CPython")
                print("  (This is expected for the prototype implementation)")
            
            if pypy_stats:
                comparison = pycode_rl_stats['mean_speedup'] / pypy_stats['mean_speedup']
                if comparison > 1.0:
                    print(f"• PyCodeRL outperformed PyPy by {comparison:.2f}×")
                else:
                    print(f"• PyPy outperformed PyCodeRL by {1/comparison:.2f}×")
        
        print("• Results demonstrate the potential of RL-based compilation")
        print("• Further training and optimization could improve performance")

class ContinuousEvaluator:
    """Continuous evaluation during training"""
    
    def __init__(self, benchmark_suite: BenchmarkSuite):
        self.benchmark_suite = benchmark_suite
        self.runner = PythonImplementationRunner()
        self.history = []
    
    def evaluate_checkpoint(self, checkpoint_path: str) -> Dict:
        """Evaluate a training checkpoint"""
        
        # Load checkpoint
        self.runner.pycode_rl_compiler.load_checkpoint(checkpoint_path)
        
        # Run subset of benchmarks
        eval_benchmarks = ['fibonacci', 'factorial', 'list_operations', 'quicksort']
        
        results = []
        for benchmark_name in eval_benchmarks:
            if benchmark_name in self.benchmark_suite.benchmarks:
                code = self.benchmark_suite.benchmarks[benchmark_name]
                benchmark_results = self.runner.run_benchmark(
                    benchmark_name, code, ['cpython', 'pycode_rl'], iterations=1
                )
                results.extend(benchmark_results)
        
        # Calculate metrics
        pycode_rl_results = [r for r in results if r.implementation == 'pycode_rl']
        cpython_results = [r for r in results if r.implementation == 'cpython']
        
        avg_speedup = 0
        success_rate = 0
        
        if pycode_rl_results and cpython_results:
            speedups = []
            successes = 0
            
            for pycode_result in pycode_rl_results:
                # Find corresponding CPython result
                cpython_result = next(
                    (r for r in cpython_results if r.name == pycode_result.name), 
                    None
                )
                
                if (cpython_result and 
                    pycode_result.execution_time > 0 and 
                    pycode_result.correctness_score > 0.5):
                    speedup = cpython_result.execution_time / pycode_result.execution_time
                    speedups.append(speedup)
                    successes += 1
            
            avg_speedup = statistics.mean(speedups) if speedups else 0
            success_rate = successes / len(pycode_rl_results)
        
        evaluation = {
            'checkpoint': checkpoint_path,
            'avg_speedup': avg_speedup,
            'success_rate': success_rate,
            'timestamp': time.time()
        }
        
        self.history.append(evaluation)
        return evaluation
    
    def plot_training_progress(self):
        """Plot training progress over time"""
        if not self.history:
            return
        
        checkpoints = [h['checkpoint'] for h in self.history]
        speedups = [h['avg_speedup'] for h in self.history]
        success_rates = [h['success_rate'] for h in self.history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Speedup over training
        ax1.plot(range(len(speedups)), speedups, 'b-o', label='Average Speedup')
        ax1.set_xlabel('Checkpoint')
        ax1.set_ylabel('Speedup over CPython')
        ax1.set_title('Performance Improvement During Training')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Success rate over training
        ax2.plot(range(len(success_rates)), success_rates, 'r-o', label='Success Rate')
        ax2.set_xlabel('Checkpoint')
        ax2.set_ylabel('Compilation Success Rate')
        ax2.set_title('Compilation Reliability During Training')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        traceback.print_exc()