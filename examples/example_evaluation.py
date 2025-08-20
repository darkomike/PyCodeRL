#!/usr/bin/env python3
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
    print("\nPerformance Summary:")
    for impl, stats in report.get('summary', {}).items():
        print(f"{impl}: {stats['mean_speedup']:.2f}x average speedup")
    
    # Create visualizations
    analyzer.plot_performance_comparison()

if __name__ == "__main__":
    main()
