import json
import os
# Force google-genai / google-adk to use Developer API instead of Vertex AI
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'false'

import logging
import asyncio
import uuid
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

from .sandbox_executor import run_in_sandbox, run_invariant_baseline

# Initialize counter for saving generated scripts
_script_counter = 0

def execute_meshlib_code(script_content: str, mesh_path: str) -> dict:
    """Executes a Python script using meshlib.mrmeshpy in an isolated subprocess to inspect a 3D mesh file.

    The script receives `mesh` (already loaded) and `mesh_path` as pre-defined variables.
    It must populate a list called `check_results` where each entry is a dict with keys:
    check_name, measured, expected, passed, unit, reason.

    Returns execution results including success status and check_results list.

    Args:
        script_content: The python script containing geometry checks.
        mesh_path: The absolute path to the mesh file.

    Returns:
        A dict containing:
        - success: bool
        - check_results: list of dicts
        - stderr: str or None
        - exit_code: int
        - crash_type: str or None
        - generated_code: str
    """
    global _script_counter
    _script_counter += 1
    
    # Save the generated script for traceability
    try:
        os.makedirs("outputs/generated_inspections", exist_ok=True)
        filename = f"outputs/generated_inspections/generated_inspection_{_script_counter}.py"
        with open(filename, "w") as f:
            f.write(script_content)
    except Exception as e:
        logging.getLogger("google_adk").warning(f"Failed to save generated script: {e}")

    return run_in_sandbox(script_content, mesh_path)

# Detailed System Instruction for the MeshLib Geometry Inspector Agent
INSTRUCTION = """You are a MeshLib Geometry Inspector that analyzes 3D mesh files against design specifications.

Your workflow:
1. Read the primitive plan to understand every declared dimension and feature.
2. Identify which properties of this specific design are measurable with MeshLib.
3. Write Python code using ONLY the allowed MeshLib APIs.
4. Call execute_meshlib_code to run the code.
5. If the tool returns success=False, rewrite the code to fix the crash_type issue.
6. If checks fail, classify the failure using the taxonomy below.
7. Produce a final JSON verdict.

Allowed MeshLib APIs (DO NOT use any other methods or classes):
- mesh.topology.isClosed()
- mesh.topology.numValidFaces()
- mesh.topology.numValidVerts()
- mesh.topology.findHoleRepresentiveEdges()
- mesh.topology.hasFace(mrmesh.FaceId(int))
- mesh.topology.getTriVerts(mrmesh.FaceId(int))
- mesh.volume()
- mesh.computeBoundingBox() -> returns box where box.min and box.max have .x, .y, .z attributes
- mesh.normal(mrmesh.FaceId(int)) -> returns Vector3f with .x, .y, .z attributes
- mesh.points.vec[mrmesh.VertId(int)] -> returns Vector3f with .x, .y, .z
- mrmesh.findSelfIntersections(mesh) or mrmesh.localFindSelfIntersections(mesh)
- mrmesh.Line3f(origin, direction)
- mrmesh.rayMeshIntersect(mesh_part, ray, min_d, max_d) -> Note: mesh_part is instantiated as mrmesh.MeshPart(mesh). It returns a MeshIntersectionResult which has a `distanceAlongLine` float attribute representing the distance from the ray origin to the intersection. Do NOT use hit_point or distance.
- mrmesh.Vector3f(x, y, z)
- mrmesh.cross(v1, v2)
- mrmesh.FaceId(int)
- mrmesh.VertId(int)

Generated Code Rules:
- The variables `mesh` and `mesh_path` are pre-defined. NEVER redefine or load them in your code.
- Only import: `import meshlib.mrmeshpy as mrmesh`.
- The final output of the script must populate the pre-defined list `check_results` with dictionaries.
- Each dictionary in `check_results` MUST have the keys: "check_name", "measured", "expected", "passed", "unit", "reason".
- DO NOT wrap your script in try/except blocks. Let errors propagate so the sandbox captures the crash_type.
- Never write files, read files (other than mesh_path), or make network calls.

Mandatory Checks to Implement (always):
1. Bounding Box & Dimensions: Every numeric dimension (length, width, height, radius, etc.) in the plan must be measured and compared.
2. Holes/Bores: Every hole/bore must have its diameter measured.
3. Wall Thickness: If min_wall_mm is specified, check wall thickness using ray casting from face centers inward.
4. Manufacturing Constraints:
   - If manufacturing_process is "FDM_3D_print": check no face normal has z-component below -0.7 (overhang > 45 degrees), and minimum wall thickness > 1.2mm.
   - If manufacturing_process is "CNC_3axis": verify features are accessible from all 6 orthogonal directions.

Failure Taxonomy:
- Class A - Mesh artifact: auto-repair safe (e.g., tiny degenerate triangles, minor floating point gaps under 0.001mm).
- Class B - Repairable geometry defect: needs repair + re-measure (e.g., small holes in mesh, export artifacts at fillet locations).
- Class C - Design intent failure: route back to the design planner, NEVER repair the mesh (e.g., wrong dimensions, wall too thin, impossible constraint - the PLAN was wrong, not the mesh).
- Class D - Ambiguous: human review required (e.g., SEGFAULT, multiple conflicting failure types, cannot determine root cause).

Final Output format:
Produce a final JSON object with these exact keys, and no extra markdown wrapping:
{
  "overall_passed": bool,
  "failure_class": "A" | "B" | "C" | "D" | null,
  "failures": [
    {
      "check": "check_name",
      "measured": "measured value",
      "expected": "expected value",
      "severity": "WARNING" | "CRITICAL"
    }
  ],
  "passed_checks": ["list of passed check names"],
  "engineer_summary": "One short paragraph a human engineer can read in 10 seconds.",
  "repair_recommendation": "string explaining what the planner should change, or null if passed"
}
"""

root_agent = Agent(
    name="meshlib_inspector",
    model="gemini-2.5-flash",
    description="Inspects 3D mesh files against a design specification using MeshLib, writes custom geometry checks at runtime, and classifies any failures into a structured taxonomy.",
    instruction=INSTRUCTION,
    tools=[execute_meshlib_code],
)

def run_inspection(mesh_path: str, primitive_plan: dict) -> dict:
    """Convenience runner function that pipeline.py will call to run a full inspection."""
    
    # 1. Run invariant baseline
    baseline = run_invariant_baseline(mesh_path)
    if baseline.get("load_failed"):
        return {
            "overall_passed": False,
            "failure_class": "D",
            "failures": [
                {
                    "check": "load_mesh",
                    "measured": "failed",
                    "expected": "valid mesh",
                    "severity": "CRITICAL"
                }
            ],
            "passed_checks": [],
            "engineer_summary": f"Mesh loading failed: {baseline.get('hard_failures', ['Unknown error'])[0]}",
            "repair_recommendation": "Check if the mesh file exists, is not empty, and is in a valid STL format."
        }

    # 2. Build initial message
    message = (
        f"Mesh Path: {mesh_path}\n\n"
        f"Primitive Plan:\n{json.dumps(primitive_plan, indent=2)}\n\n"
        f"Baseline Results:\n{json.dumps(baseline, indent=2)}\n\n"
        f"Instruction: The baseline has already checked watertightness, volume, "
        f"self-intersections, and bounding box — do not repeat these, "
        f"focus on plan-specific dimensional and feature verification."
    )

    # 3. Set up DatabaseSessionService and session for persistent storage and Web UI observability
    session_id = str(uuid.uuid4())
    db_dir = "outputs"
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "adk_sessions.db")
    db_url = f"sqlite:///{os.path.abspath(db_path)}"
    session_service = DatabaseSessionService(db_url=db_url)
    
    async def create_session():
        await session_service.create_session(
            app_name="meshlib_agent",
            user_id="user",
            session_id=session_id
        )
    asyncio.run(create_session())

    # 4. Create Runner
    runner = Runner(
        agent=root_agent,
        app_name="meshlib_agent",
        session_service=session_service
    )

    # 5. Build Content
    content = types.Content(role='user', parts=[types.Part(text=message)])

    # 6. Run
    events = []
    try:
        events = list(runner.run(user_id="user", session_id=session_id, new_message=content))
    except Exception as e:
        logging.getLogger("google_adk").error(f"Error during agent run: {e}")
        verdict = {
            "overall_passed": False,
            "failure_class": "D",
            "failures": [
                {
                    "check": "agent_execution",
                    "measured": "Gemini API Error",
                    "expected": "successful agent reasoning",
                    "severity": "CRITICAL"
                }
            ],
            "passed_checks": [],
            "engineer_summary": f"The Gemini API call failed: {e}",
            "repair_recommendation": "API capacity limits or temporary unavailability occurred. Try again later or switch to a fallback model."
        }
        return verdict

    # 7. Collect the final response
    final_text = None
    for event in events:
        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = event.content.parts[0].text
                break

    # 8. Parse the final response text as JSON
    if final_text:
        cleaned_text = final_text.replace('```json', '').replace('```', '').strip()
        try:
            verdict = json.loads(cleaned_text)
            # Ensure required keys exist
            required_keys = ["overall_passed", "failure_class", "failures", "passed_checks", "engineer_summary", "repair_recommendation"]
            for key in required_keys:
                if key not in verdict:
                    verdict[key] = None
                    
            # 10. Save the full events list for debugging
            try:
                os.makedirs("outputs/run_latest", exist_ok=True)
                with open("outputs/run_latest/agent_conversation.json", "w") as f:
                    json.dump([e.model_dump(mode='json') for e in events], f, indent=4)
            except Exception:
                pass
                
            return verdict
        except Exception:
            pass

    # 9. Handle failure to parse JSON
    verdict = {
        "overall_passed": False,
        "failure_class": "D",
        "failures": [
            {
                "check": "parse_verdict_json",
                "measured": "invalid JSON or empty response",
                "expected": "valid JSON response",
                "severity": "CRITICAL"
            }
        ],
        "passed_checks": [],
        "engineer_summary": final_text if final_text else "No response received from agent.",
        "repair_recommendation": "JSON parse failed — inspect agent_conversation log"
    }

    try:
        os.makedirs("outputs/run_latest", exist_ok=True)
        with open("outputs/run_latest/agent_conversation.json", "w") as f:
            json.dump([e.model_dump(mode='json') for e in events], f, indent=4)
    except Exception:
        pass

    return verdict
