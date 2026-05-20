import cadquery as cq
from src.logger import get_agent_logger

logger = get_agent_logger()

def execute_cad_code(code: str):
    """Executes a string of Python code and returns the `result_solid` object."""
    logger.info("Setting up local scope for CadQuery execution...")
    local_scope = {}
    try:
        # Execute the generated code with cadquery available in its scope
        exec(code, {"cq": cq}, local_scope)
        solid = local_scope.get("result_solid")
        
        if solid is None:
            raise ValueError("The generated code did not produce a 'result_solid' variable.")
            
        return solid
    except Exception as e:
        raise RuntimeError(f"Code execution failed: {e}")

def export_solid(solid, filename: str):
    """Exports a CadQuery solid to the specified filename (e.g., .step or .stl)."""
    cq.exporters.export(solid, filename)
    return filename
