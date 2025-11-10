import pickle
import sys
import types
from collections.abc import Iterable

def analyze_object_size(obj, name="root", max_depth=5, current_depth=0,
                       min_size_mb=0.1, visited=None, show_all=False):
    """
    Recursively analyze object sizes and print a hierarchical breakdown.

    Parameters:
    -----------
    obj : any
        Object to analyze
    name : str
        Name/path of the current object
    max_depth : int
        Maximum recursion depth
    current_depth : int
        Current recursion depth
    min_size_mb : float
        Minimum size in MB to display (set to 0 to show everything)
    visited : set
        Set of already visited object IDs (to prevent infinite recursion)
    show_all : bool
        If True, show all attributes regardless of size
    """
    if visited is None:
        visited = set()

    # Prevent infinite recursion
    obj_id = id(obj)
    if obj_id in visited or current_depth > max_depth:
        return 0
    visited.add(obj_id)

    # Calculate size of current object
    try:
        obj_size_bytes = len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        obj_size_mb = obj_size_bytes / (1024 * 1024)
    except Exception as e:
        obj_size_mb = 0
        if show_all or current_depth == 0:
            print(f"{'  ' * current_depth}{name}: <unpicklable: {type(obj).__name__}> - {str(e)[:50]}")
        return 0

    # Print current object info if it meets size threshold
    if obj_size_mb >= min_size_mb or show_all or current_depth == 0:
        size_str = f"{obj_size_mb:.2f} MB" if obj_size_mb >= 1 else f"{obj_size_bytes:.0f} bytes"
        type_name = type(obj).__name__

        # Add warning for large objects
        warning = " ⚠️  LARGE!" if obj_size_mb > 10 else ""
        print(f"{'  ' * current_depth}{name}: {size_str} ({type_name}){warning}")

    # Don't recurse further if object is too small or we're at max depth
    if obj_size_mb < min_size_mb and current_depth > 0:
        return obj_size_mb

    if current_depth >= max_depth:
        return obj_size_mb

    # Recursively analyze attributes for objects that have them
    if hasattr(obj, '__dict__'):
        for attr_name, attr_value in obj.__dict__.items():
            if not attr_name.startswith('_'):  # Skip private attributes
                full_name = f"{name}.{attr_name}"
                analyze_object_size(attr_value, full_name, max_depth,
                                  current_depth + 1, min_size_mb, visited, show_all)

    # Handle special cases
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            if i > 10:  # Limit list inspection to first 10 items
                remaining = len(obj) - i
                print(f"{'  ' * (current_depth + 1)}{name}[{i}...{len(obj)-1}]: ... (and {remaining} more items)")
                break
            full_name = f"{name}[{i}]"
            analyze_object_size(item, full_name, max_depth,
                              current_depth + 1, min_size_mb, visited, show_all)

    elif isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, (str, int, float)):
                full_name = f"{name}['{key}']" if isinstance(key, str) else f"{name}[{key}]"
                analyze_object_size(value, full_name, max_depth,
                                  current_depth + 1, min_size_mb, visited, show_all)

    return obj_size_mb

def find_large_objects(obj, threshold_mb=1.0, max_depth=10):
    """
    Find all objects larger than threshold within the given object.

    Returns a list of (path, size_mb, type) tuples.
    """
    large_objects = []

    def _find_recursive(current_obj, path="root", depth=0, visited=None):
        if visited is None:
            visited = set()

        obj_id = id(current_obj)
        if obj_id in visited or depth > max_depth:
            return
        visited.add(obj_id)

        try:
            size_bytes = len(pickle.dumps(current_obj))
            size_mb = size_bytes / (1024 * 1024)

            if size_mb >= threshold_mb:
                large_objects.append((path, size_mb, type(current_obj).__name__))

            # Recurse into attributes
            if hasattr(current_obj, '__dict__'):
                for attr_name, attr_value in current_obj.__dict__.items():
                    if not attr_name.startswith('_'):
                        _find_recursive(attr_value, f"{path}.{attr_name}", depth + 1, visited)

            # Handle collections
            if isinstance(current_obj, (list, tuple)):
                for i, item in enumerate(current_obj[:5]):  # Limit to first 5 items
                    _find_recursive(item, f"{path}[{i}]", depth + 1, visited)
            elif isinstance(current_obj, dict):
                for key, value in list(current_obj.items())[:5]:  # Limit to first 5 items
                    if isinstance(key, (str, int)):
                        key_str = f"'{key}'" if isinstance(key, str) else str(key)
                        _find_recursive(value, f"{path}[{key_str}]", depth + 1, visited)

        except Exception:
            pass

    _find_recursive(obj)

    # Sort by size (largest first)
    large_objects.sort(key=lambda x: x[1], reverse=True)
    return large_objects

def quick_calibration_diagnosis(calibration_obj):
    """
    Quick diagnosis specifically for calibration objects.
    """
    print("="*80)
    print("QUICK CALIBRATION OBJECT DIAGNOSIS")
    print("="*80)

    if calibration_obj is None:
        print("Calibration object is None")
        return

    # Check overall size
    try:
        total_size = len(pickle.dumps(calibration_obj)) / (1024 * 1024)
        print(f"Total calibration object size: {total_size:.2f} MB")
    except Exception as e:
        print(f"Cannot calculate total size: {e}")
        return

    print("\nTop-level attributes:")
    for attr_name in dir(calibration_obj):
        if not attr_name.startswith('_'):
            try:
                attr_value = getattr(calibration_obj, attr_name)
                if not callable(attr_value):
                    attr_size = len(pickle.dumps(attr_value)) / (1024 * 1024)
                    if attr_size > 0.01:  # Show attributes > 10KB
                        print(f"  {attr_name}: {attr_size:.3f} MB ({type(attr_value).__name__})")
            except Exception as e:
                print(f"  {attr_name}: <error: {str(e)[:30]}>")

    # Check calibrated_models specifically
    if hasattr(calibration_obj, 'calibrated_models'):
        models = calibration_obj.calibrated_models
        if models:
            print(f"\nCalibrated models ({len(models)} total):")
            for i, model in enumerate(models):
                try:
                    model_size = len(pickle.dumps(model)) / (1024 * 1024)
                    print(f"  Model {i}: {model_size:.3f} MB ({type(model).__name__})")

                    # Check if model has large attributes
                    if model_size > 1.0:
                        print(f"    WARNING: Model {i} is large! Checking attributes...")
                        for attr in ['calibrators_', 'base_estimator', 'estimators']:
                            if hasattr(model, attr):
                                try:
                                    attr_val = getattr(model, attr)
                                    attr_size = len(pickle.dumps(attr_val)) / (1024 * 1024)
                                    if attr_size > 0.1:
                                        print(f"      {attr}: {attr_size:.3f} MB")
                                except:
                                    pass
                except Exception as e:
                    print(f"  Model {i}: <error calculating size: {e}>")

    print("="*80)

# Usage examples:

def analyze_object(obj):
    """
    Complete analysis of your calibration object.
    """
    print("STEP 1: Quick Diagnosis")
    quick_calibration_diagnosis(obj)

    print("\nSTEP 2: Find Large Objects (>1MB)")
    large_objects = find_large_objects(obj, threshold_mb=1.0)
    if large_objects:
        for path, size_mb, obj_type in large_objects:
            print(f"  {path}: {size_mb:.2f} MB ({obj_type})")
    else:
        print("  No objects larger than 1MB found")

    print("\nSTEP 3: Detailed Recursive Analysis")
    print("(Showing objects larger than 0.1MB)")
    analyze_object_size(obj, "calibration", max_depth=4, min_size_mb=0.1)

    print("\nSTEP 4: Show ALL attributes (top 2 levels)")
    analyze_object_size(obj, "calibration", max_depth=2, min_size_mb=0, show_all=True)

# To use this on your calibration object:
# analyze_object(pl.calibration):

# To use this on your pipeline object
# analyze_object(pl):