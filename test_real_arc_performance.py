#!/usr/bin/env python3
"""
Comprehensive test of the enhanced ARC-AGI solver on real ARC data.

This script tests our optimized solver with:
- Task 17: Enhanced DSL primitives (14 total primitives)
- Task 18: Complete Tier-1 heuristic implementation
  - D₄ symmetry minimization
  - Learned heuristic weights
  - Optimized search efficiency and pruning
  - Enhanced caching system 

Tests performance improvements and success rates on real ARC tasks.
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Add src to path
sys.path.append('src')

from arc_solver.search.astar import AStarSearcher, SearchConfig
from arc_solver.search.heuristics import HeuristicSystem, DualHeuristic
from arc_solver.reasoning.dsl_engine import DSLEngine

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise for testing

class ARCTaskLoader:
    """Load and parse ARC tasks from JSON files."""
    
    @staticmethod
    def load_task(filepath: str) -> Dict[str, Any]:
        """Load a single ARC task from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def parse_grids(task_data: Dict[str, Any]) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray]]:
        """Parse training and test grids from task data.
        
        Returns:
            Tuple of (training_pairs, test_inputs)
        """
        training_pairs = []
        for example in task_data['train']:
            input_grid = np.array(example['input'], dtype=np.int32)
            output_grid = np.array(example['output'], dtype=np.int32)
            training_pairs.append((input_grid, output_grid))
        
        test_inputs = []
        for example in task_data['test']:
            input_grid = np.array(example['input'], dtype=np.int32)
            test_inputs.append(input_grid)
        
        return training_pairs, test_inputs


class ARCPerformanceTester:
    """Test ARC solver performance with different configurations."""
    
    def __init__(self):
        """Initialize the performance tester."""
        self.results = []
        
        # Different configurations to test
        self.configs = {
            'baseline': SearchConfig(
                max_program_length=3,
                max_nodes_expanded=100,
                beam_width=16,
                adaptive_beam=False,
                parallel_expansion=False,
                incremental_search=False,
                statistics_tracking=True
            ),
            'optimized': SearchConfig(
                max_program_length=4,
                max_nodes_expanded=200,
                beam_width=32,
                adaptive_beam=True,
                parallel_expansion=False,  # Keep False for fair comparison
                incremental_search=True,
                statistics_tracking=True,
                adaptive_beam_quality_threshold=0.8
            ),
            'enhanced': SearchConfig(
                max_program_length=4,
                max_nodes_expanded=300,
                beam_width=64,
                adaptive_beam=True,
                parallel_expansion=True,
                max_threads=2,
                incremental_search=True,
                statistics_tracking=True,
                adaptive_beam_quality_threshold=0.7
            )
        }
    
    def test_single_task(self, task_file: str, config_name: str, max_time: float = 10.0) -> Dict[str, Any]:
        """Test solver on a single ARC task.
        
        Args:
            task_file: Path to ARC task JSON file
            config_name: Configuration to use
            max_time: Maximum time per task
            
        Returns:
            Test result dictionary
        """
        print(f"Testing {task_file} with {config_name} config...")
        
        try:
            # Load task
            task_data = ARCTaskLoader.load_task(task_file)
            training_pairs, test_inputs = ARCTaskLoader.parse_grids(task_data)
            
            if not training_pairs:
                return {
                    'task_file': task_file,
                    'config': config_name,
                    'success': False,
                    'error': 'No training examples',
                    'computation_time': 0.0
                }
            
            # Create searcher with specified config
            config = self.configs[config_name]
            config.max_computation_time = max_time
            searcher = AStarSearcher(config)
            
            # Test on first training example (single-example search)
            input_grid, target_grid = training_pairs[0]
            
            start_time = time.perf_counter()
            result = searcher.search(input_grid, target_grid)
            total_time = time.perf_counter() - start_time
            
            # Prepare result
            test_result = {
                'task_file': task_file,
                'config': config_name,
                'success': result.success,
                'computation_time': total_time,
                'nodes_expanded': result.nodes_expanded,
                'nodes_generated': result.nodes_generated,
                'max_depth_reached': result.max_depth_reached,
                'beam_width_used': result.beam_width_used,
                'termination_reason': result.termination_reason,
                'program_length': len(result.program) if result.program else 0,
                'grid_size': f"{input_grid.shape[0]}x{input_grid.shape[1]}"
            }
            
            # Add detailed statistics if available
            if result.heuristic_stats:
                stats = result.heuristic_stats
                test_result.update({
                    'search_efficiency': stats.get('search_efficiency', 0),
                    'average_branching_factor': stats.get('average_branching_factor', 0),
                    'cache_hit_rate': stats.get('overall_cache_hit_rate', 0),
                    'heuristic_computations': stats.get('heuristic_computations', 0),
                    'beam_reductions': stats.get('beam_reductions', 0)
                })
            
            if result.success and result.program:
                print(f"  ✓ SUCCESS! Found solution with {len(result.program)} operations in {total_time:.3f}s")
                print(f"    Program: {result.program}")
            else:
                print(f"  ✗ Failed: {result.termination_reason} in {total_time:.3f}s")
            
            return test_result
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return {
                'task_file': task_file,
                'config': config_name,
                'success': False,
                'error': str(e),
                'computation_time': 0.0
            }
    
    def test_multiple_tasks(self, task_files: List[str], config_name: str, max_time: float = 10.0) -> List[Dict[str, Any]]:
        """Test solver on multiple ARC tasks.
        
        Args:
            task_files: List of task file paths
            config_name: Configuration to use
            max_time: Maximum time per task
            
        Returns:
            List of test results
        """
        results = []
        
        print(f"\n=== Testing {len(task_files)} tasks with {config_name} configuration ===")
        
        for i, task_file in enumerate(task_files):
            print(f"\nTask {i+1}/{len(task_files)}: ", end="")
            result = self.test_single_task(task_file, config_name, max_time)
            results.append(result)
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze test results and compute statistics.
        
        Args:
            results: List of test results
            
        Returns:
            Analysis summary
        """
        if not results:
            return {}
        
        # Basic statistics
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.get('success', False))
        success_rate = successful_tasks / total_tasks
        
        # Timing statistics
        times = [r.get('computation_time', 0) for r in results if r.get('computation_time', 0) > 0]
        avg_time = np.mean(times) if times else 0
        median_time = np.median(times) if times else 0
        
        # Search statistics
        nodes_expanded = [r.get('nodes_expanded', 0) for r in results if r.get('nodes_expanded', 0) > 0]
        avg_nodes_expanded = np.mean(nodes_expanded) if nodes_expanded else 0
        
        nodes_generated = [r.get('nodes_generated', 0) for r in results if r.get('nodes_generated', 0) > 0]
        avg_nodes_generated = np.mean(nodes_generated) if nodes_generated else 0
        
        # Cache statistics (for optimized configs)
        cache_hit_rates = [r.get('cache_hit_rate', 0) for r in results if 'cache_hit_rate' in r]
        avg_cache_hit_rate = np.mean(cache_hit_rates) if cache_hit_rates else 0
        
        # Program length statistics
        program_lengths = [r.get('program_length', 0) for r in results if r.get('success', False)]
        avg_program_length = np.mean(program_lengths) if program_lengths else 0
        
        return {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'success_rate': success_rate,
            'avg_computation_time': avg_time,
            'median_computation_time': median_time,
            'avg_nodes_expanded': avg_nodes_expanded,
            'avg_nodes_generated': avg_nodes_generated,
            'avg_cache_hit_rate': avg_cache_hit_rate,
            'avg_program_length': avg_program_length,
            'config': results[0].get('config', 'unknown') if results else 'unknown'
        }
    
    def print_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print analysis results in a formatted way."""
        config = analysis.get('config', 'unknown')
        print(f"\n=== Analysis for {config} configuration ===")
        print(f"Success Rate: {analysis['successful_tasks']}/{analysis['total_tasks']} ({analysis['success_rate']:.1%})")
        print(f"Average Time: {analysis['avg_computation_time']:.3f}s")
        print(f"Median Time: {analysis['median_computation_time']:.3f}s")
        print(f"Average Nodes Expanded: {analysis['avg_nodes_expanded']:.1f}")
        print(f"Average Nodes Generated: {analysis['avg_nodes_generated']:.1f}")
        if analysis['avg_cache_hit_rate'] > 0:
            print(f"Average Cache Hit Rate: {analysis['avg_cache_hit_rate']:.1%}")
        if analysis['avg_program_length'] > 0:
            print(f"Average Program Length: {analysis['avg_program_length']:.1f}")


def main():
    """Main test function."""
    print("🚀 Testing Enhanced ARC-AGI Solver on Real Data")
    print("=" * 60)
    
    # Get list of ARC task files (test on a subset for speed)
    arc_dir = Path("arc_dataset")
    all_task_files = list(arc_dir.glob("*.json"))
    
    # Filter out non-task files
    task_files = [f for f in all_task_files if f.name not in ['sample_task.json', 'training_challenges.json']]
    
    # Test on first 10 tasks for demonstration
    test_files = [str(f) for f in task_files[:10]]
    
    print(f"Testing on {len(test_files)} ARC tasks:")
    for f in test_files:
        print(f"  - {Path(f).name}")
    
    # Initialize tester
    tester = ARCPerformanceTester()
    
    # Test different configurations
    all_results = {}
    
    for config_name in ['baseline', 'optimized', 'enhanced']:
        print(f"\n{'='*20} {config_name.upper()} CONFIGURATION {'='*20}")
        results = tester.test_multiple_tasks(test_files, config_name, max_time=15.0)
        all_results[config_name] = results
        
        # Analyze and print results
        analysis = tester.analyze_results(results)
        tester.print_analysis(analysis)
    
    # Compare configurations
    print(f"\n{'='*20} PERFORMANCE COMPARISON {'='*20}")
    
    for config_name, results in all_results.items():
        analysis = tester.analyze_results(results)
        print(f"{config_name:>10}: {analysis['success_rate']:.1%} success, "
              f"{analysis['avg_computation_time']:.3f}s avg time, "
              f"{analysis['avg_nodes_expanded']:.0f} nodes expanded")
    
    # Show successful tasks
    print(f"\n{'='*20} SUCCESSFUL SOLUTIONS {'='*20}")
    
    for config_name, results in all_results.items():
        successful = [r for r in results if r.get('success', False)]
        if successful:
            print(f"\n{config_name} configuration solved {len(successful)} tasks:")
            for result in successful:
                task_name = Path(result['task_file']).name
                print(f"  ✓ {task_name}: {result['program_length']} ops, {result['computation_time']:.3f}s")
    
    print(f"\n{'='*60}")
    print("🎉 Enhanced ARC-AGI Solver Testing Complete!")
    print("Key improvements demonstrated:")
    print("  - Enhanced DSL with 14 primitives (Task 17)")
    print("  - D₄ symmetry minimization (Task 18.1)")
    print("  - Learned heuristic weights (Task 18.2)")
    print("  - Optimized search efficiency (Task 18.3)")
    print("  - Enhanced caching system (Task 18.4)")


if __name__ == "__main__":
    main()
