import sys
import traceback

# Add src to path
sys.path.append('.')

def test_import(module_name):
    print(f"\n{'='*60}")
    print(f"Testing import: {module_name}")
    print('='*60)
    try:
        __import__(module_name)
        print(f"✓ {module_name} imported successfully")
    except Exception as e:
        print(f"✗ {module_name} import failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\nException details:")
        print(f"  Type: {type(e).__name__}")
        print(f"  Message: {str(e)}")

        # Try to get more context about where the error occurred
        tb = sys.exc_info()[2]
        while tb.tb_next:
            tb = tb.tb_next
        frame = tb.tb_frame
        print(f"\nError occurred in:")
        print(f"  File: {frame.f_code.co_filename}")
        print(f"  Function: {frame.f_code.co_name}")
        print(f"  Line: {tb.tb_lineno}")

        # Print local variables at the error point
        print("\nLocal variables at error point:")
        for var_name, var_value in frame.f_locals.items():
            if not var_name.startswith('__'):
                print(f"  {var_name} = {repr(var_value)[:100]}...")

# Test individual imports
print("Testing individual module imports...")

# First test the flow models
test_import('flow_models.wolf.flows.flow')
test_import('flow_models.wolf.flows.activation')
test_import('flow_models.wolf.flows')

# Then test the main modules
test_import('datasets')
test_import('models.ddpm')
test_import('evaluation')
test_import('modern_metrics')
test_import('run_lib')
test_import('op.fused_act')
