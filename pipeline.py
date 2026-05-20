import os
import datetime
from src.logger import get_agent_logger
from src.llm import generate_cad_code, extract_expected_dimensions
from src.cad_executor import execute_cad_code, export_solid
from src.mesh_inspector import run_all_inspections

def run_pipeline(request_prompt: str, output_base_dir: str = "outputs"):
    # 1. Setup output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Setup Agent Logger
    log_file = os.path.join(output_dir, "pipeline.log")
    logger = get_agent_logger(log_file)
    
    logger.info(f"--- Starting Advanced Agentic CAD Pipeline ---")
    logger.info(f"Output directory initialized: {output_dir}")
    
    # Calculate Expected Dimensions Dynamically
    expected_dimensions = extract_expected_dimensions(request_prompt)
    
    # 3 & 4. Generate and Execute Code with Retry Logic
    MAX_RETRIES = 3
    solid = None
    current_prompt = request_prompt
    
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"--- Attempt {attempt}/{MAX_RETRIES} ---")
        logger.info("[1/4] Generating code via LLM...")
        try:
            code = generate_cad_code(current_prompt)
            
            # Save generated code
            code_path = os.path.join(output_dir, f"generated_code_attempt_{attempt}.py")
            with open(code_path, "w") as f:
                f.write(code)
            logger.info(f"Code successfully generated and saved to {code_path}")
            
            logger.info("[2/4] Executing generated CAD code...")
            solid = execute_cad_code(code)
            logger.info("Solid generated successfully in-memory.")
            break # Success, break out of retry loop
            
        except Exception as e:
            logger.error(f"Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                logger.info("Retrying with error feedback...")
                current_prompt = (
                    f"The previous code generated the following error during execution:\n"
                    f"```\n{str(e)}\n```\n"
                    f"Please fix the code. Remember to output ONLY valid Python code. "
                    f"Original request: {request_prompt}"
                )
            else:
                logger.error("Max retries reached. Pipeline failed.")
                return
        
    # 5. Export artifacts
    logger.info("[3/4] Exporting solid to STEP and STL...")
    step_filename = os.path.join(output_dir, "model.step")
    stl_filename = os.path.join(output_dir, "model.stl")
    
    try:
        export_solid(solid, step_filename)
        export_solid(solid, stl_filename)
        logger.info(f"Exported STEP to {step_filename}")
        logger.info(f"Exported STL to {stl_filename}")
    except Exception as e:
        logger.error(f"Failed to export models: {e}")
        return
    
    # 6. Advanced Inspection
    logger.info("[4/4] Running advanced MeshLib inspections...")
    try:
        report = run_all_inspections(stl_filename, expected_dimensions, output_dir)
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Inspection error: {e}")

if __name__ == "__main__":
    # ULTIMATE STRESS TEST: Centrifugal Compressor Impeller
    # This is one of the most notoriously difficult geometries to generate correctly with code,
    # involving lofts, sweeps along curved paths, and conical boolean intersections.
    test_prompt = (
        "Create a complex Centrifugal Compressor Impeller. "
        "1. The main hub is a truncated cone with a base diameter of 100mm (at Z=0), a top diameter of 30mm (at Z=60), and a total height of 60mm. "
        "2. The hub has a central bore hole of 15mm diameter going all the way through the Z axis for the driveshaft. "
        "3. On the surface of the hub, create 7 swept curved aerodynamic blades. "
        "4. Each blade should start at the base (radius 50mm) and curve upwards along the surface of the cone to the top (radius 15mm). "
        "5. The blades should have a uniform thickness of 2mm, an outward protrusion (height off the hub surface) of 15mm at the base, tapering to 5mm at the top. "
        "6. The blades should curve/twist around the Z axis by roughly 60 degrees from bottom to top to create the aerodynamic impeller shape. "
        "7. Ensure the final object is a single unified solid, assigned to a variable named 'result_solid'. "
        "DO NOT use hallucinated Selectors, stick to standard CadQuery operations like workplanes, extrude, sweep, or loft."
    )
    
    run_pipeline(test_prompt)