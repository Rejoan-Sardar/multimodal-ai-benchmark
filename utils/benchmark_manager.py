"""
Benchmark Manager for creating, saving, and managing custom benchmarks
"""

import os
import json
import uuid
import time
from datetime import datetime

# Directory for storing custom benchmarks
CUSTOM_BENCHMARKS_DIR = "custom_benchmarks"
os.makedirs(CUSTOM_BENCHMARKS_DIR, exist_ok=True)

def generate_benchmark_id():
    """Generate a unique ID for a new benchmark"""
    return f"custom_{int(time.time())}_{str(uuid.uuid4())[:8]}"

def create_benchmark(name, description, modality_type, metrics, example_data=None, task_description=None):
    """
    Create a new custom benchmark
    
    Args:
        name: Name of the benchmark
        description: Detailed description of the benchmark
        modality_type: Type of benchmark (text, image, video, audio, multimodal)
        metrics: List of metrics for evaluation
        example_data: Example data for the benchmark (optional)
        task_description: Description of the task (optional)
        
    Returns:
        dict: The created benchmark
    """
    benchmark_id = generate_benchmark_id()
    
    # Create benchmark structure
    benchmark = {
        "id": benchmark_id,
        "name": name,
        "type": modality_type,
        "description": description,
        "metrics": metrics,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_custom": True
    }
    
    # Add example if provided
    if example_data or task_description:
        benchmark["example"] = {}
        if example_data:
            benchmark["example"]["input"] = example_data
        if task_description:
            benchmark["example"]["task_description"] = task_description
    
    # Save benchmark to file
    save_benchmark(benchmark)
    
    return benchmark

def save_benchmark(benchmark):
    """
    Save a benchmark to file
    
    Args:
        benchmark: Benchmark data
    """
    benchmark_path = os.path.join(CUSTOM_BENCHMARKS_DIR, f"{benchmark['id']}.json")
    with open(benchmark_path, 'w') as f:
        json.dump(benchmark, f, indent=2)

def load_benchmarks():
    """
    Load all custom benchmarks
    
    Returns:
        list: All custom benchmarks
    """
    benchmarks = []
    
    if os.path.exists(CUSTOM_BENCHMARKS_DIR):
        for filename in os.listdir(CUSTOM_BENCHMARKS_DIR):
            if filename.endswith('.json'):
                file_path = os.path.join(CUSTOM_BENCHMARKS_DIR, filename)
                try:
                    with open(file_path, 'r') as f:
                        benchmark = json.load(f)
                        benchmarks.append(benchmark)
                except Exception as e:
                    print(f"Error loading benchmark {filename}: {e}")
    
    return benchmarks

def get_benchmark(benchmark_id):
    """
    Get a specific benchmark by ID
    
    Args:
        benchmark_id: ID of the benchmark
        
    Returns:
        dict: Benchmark data or None if not found
    """
    file_path = os.path.join(CUSTOM_BENCHMARKS_DIR, f"{benchmark_id}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading benchmark {benchmark_id}: {e}")
    
    return None

def update_benchmark(benchmark_id, updates):
    """
    Update an existing benchmark
    
    Args:
        benchmark_id: ID of the benchmark to update
        updates: Dictionary of fields to update
        
    Returns:
        dict: Updated benchmark or None if not found
    """
    benchmark = get_benchmark(benchmark_id)
    if benchmark:
        for key, value in updates.items():
            benchmark[key] = value
        
        save_benchmark(benchmark)
        return benchmark
    
    return None

def delete_benchmark(benchmark_id):
    """
    Delete a benchmark
    
    Args:
        benchmark_id: ID of the benchmark to delete
        
    Returns:
        bool: True if deleted, False otherwise
    """
    file_path = os.path.join(CUSTOM_BENCHMARKS_DIR, f"{benchmark_id}.json")
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            print(f"Error deleting benchmark {benchmark_id}: {e}")
    
    return False

def export_benchmark(benchmark_id, export_format="json"):
    """
    Export a benchmark in the specified format
    
    Args:
        benchmark_id: ID of the benchmark to export
        export_format: Format to export (json, yaml, etc.)
        
    Returns:
        str: Path to exported file or None if failed
    """
    benchmark = get_benchmark(benchmark_id)
    if not benchmark:
        return None
    
    if export_format == "json":
        export_path = os.path.join(CUSTOM_BENCHMARKS_DIR, f"export_{benchmark_id}.json")
        with open(export_path, 'w') as f:
            json.dump(benchmark, f, indent=2)
        return export_path
    
    # Add support for other formats as needed
    
    return None

def import_benchmark(file_path):
    """
    Import a benchmark from file
    
    Args:
        file_path: Path to the benchmark file
        
    Returns:
        dict: Imported benchmark or None if failed
    """
    try:
        with open(file_path, 'r') as f:
            benchmark = json.load(f)
            
            # Validate benchmark structure
            required_fields = ["name", "type", "description", "metrics"]
            for field in required_fields:
                if field not in benchmark:
                    print(f"Missing required field: {field}")
                    return None
            
            # Ensure the benchmark has a unique ID
            benchmark["id"] = generate_benchmark_id()
            benchmark["is_custom"] = True
            benchmark["imported_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            save_benchmark(benchmark)
            return benchmark
    except Exception as e:
        print(f"Error importing benchmark: {e}")
    
    return None