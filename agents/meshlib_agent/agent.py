"""
MeshLib Inspector Agent — Findings-only, no verdict power.

This agent analyzes 3D mesh files against design specifications using MeshLib.
It reports WHAT it measured and any discrepancies, but does NOT make pass/fail
decisions. The Reviewer Agent makes routing decisions.
"""

import json
import os
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'false'

import logging
import asyncio
import uuid
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

from .sandbox_executor import run_in_sandbox, run_invariant_baseline

# Globals for routing tool outputs to the correct run directory
_script_counter = 0
_current_output_dir = "outputs/run_latest"
_current_outer_attempt = 1

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
    global _script_counter, _current_output_dir, _current_outer_attempt
    _script_counter += 1
    
    # Save the generated script for traceability in the run directory
    try:
        os.makedirs(_current_output_dir, exist_ok=True)
        filename = os.path.join(_current_output_dir, f"06c_outer{_current_outer_attempt}_ai_generated_meshlib_script_{_script_counter}.py")
        with open(filename, "w") as f:
            f.write(script_content)
    except Exception as e:
        logging.getLogger("google_adk").warning(f"Failed to save generated script: {e}")

    return run_in_sandbox(script_content, mesh_path)

def explore_meshlib_api(attribute_path: str = "") -> str:
    """
    Dynamically explores the meshlib.mrmeshpy Python module.
    Use this tool when you are unsure about the exact API method name or arguments.
    
    Args:
        attribute_path: A dot-separated string representing the attribute to explore.
                        Pass an empty string "" to see top-level attributes of mrmeshpy.
                        Pass "Mesh" to see methods of the Mesh class.
                        Pass "topology" to explore topology functions.
    
    Returns:
        A string containing a list of available public attributes and their brief docstrings.
    """
    import meshlib.mrmeshpy as mrmesh
    try:
        if not attribute_path:
            obj = mrmesh
        else:
            obj = mrmesh
            for part in attribute_path.split('.'):
                obj = getattr(obj, part)
        
        attributes = dir(obj)
        public_attrs = [a for a in attributes if not a.startswith('_')]
        
        result = f"Attributes of mrmeshpy.{attribute_path if attribute_path else 'mrmeshpy'}:\n"
        
        count = 0
        for a in public_attrs:
            if count > 60:
                result += f"... and {len(public_attrs) - 60} more attributes omitted.\n"
                break
            try:
                attr_obj = getattr(obj, a)
                doc = attr_obj.__doc__ or "No docstring available."
                doc_brief = doc.strip().split('\n')[0][:120]
                result += f"- {a}: {doc_brief}\n"
            except Exception:
                result += f"- {a}\n"
            count += 1
            
        return result
    except AttributeError:
        return f"Error: Attribute '{attribute_path}' not found in meshlib.mrmeshpy."
    except Exception as e:
        return f"Error exploring module: {e}"


# System Instruction — Findings-only, NO verdict power
INSTRUCTION = """You are a MeshLib Geometry Inspector. You analyze 3D mesh files against design specifications.

CRITICAL: You report FINDINGS ONLY. You do NOT decide whether the design passes or fails.
A separate Reviewer Agent will make that decision based on your findings.

Your workflow:
1. Read the design brief to understand every declared dimension and feature.
2. Identify which properties are measurable with MeshLib.
3. If you do not know the exact MeshLib function, use `explore_meshlib_api` to discover it.
4. Write Python code using the discovered APIs.
5. Call `execute_meshlib_code` to run the code.
6. If the tool returns success=False, rewrite the code to fix the crash_type issue.
7. Report ALL findings — passed checks AND failed checks — with exact measurements.

API Discovery:
- Use `explore_meshlib_api` to dynamically search for functions, classes, and properties.
- Call with "" for top-level, "Mesh" for Mesh methods, etc.
- Always read docstrings to ensure correct argument types.

Generated Code Rules:
- `mesh` and `mesh_path` are pre-defined. NEVER redefine or load them.
- Only import: `import meshlib.mrmeshpy as mrmesh`.
- Populate the pre-defined list `check_results` with dictionaries.
- Each dict MUST have keys: "check_name", "measured", "expected", "passed", "unit", "reason".
- DO NOT wrap in try/except. Let errors propagate for crash detection.
- Never write files, read files (other than mesh_path), or make network calls.

Mandatory Checks (always implement):
1. Bounding Box & Dimensions: measure every numeric dimension from the design brief.
2. Holes/Bores: measure diameter of every declared hole/bore.
3. Wall Thickness: if min_wall_mm is specified, use ray casting from face centers inward.
4. Manufacturing Constraints:
   - FDM_3D_print: check overhang angles, minimum wall > 1.2mm.
   - CNC_3axis: verify feature accessibility from 6 orthogonal directions.

Output ONLY a JSON object with these exact keys (no markdown wrapping):
{
  "checks": [
    {
      "check_name": "string",
      "measured": "number or string",
      "expected": "number or string",
      "tolerance": "number or null",
      "passed": true/false,
      "unit": "mm or degrees or count",
      "reason": "explanation"
    }
  ],
  "anomalies": ["list of unexpected observations not covered by checks"],
  "engineer_summary": "One paragraph a human engineer can read in 10 seconds",
  "confidence": "HIGH | MEDIUM | LOW"
}
"""

root_agent = Agent(
    name="meshlib_inspector",
    model="gemini-3.1-pro-preview",
    description="Inspects 3D mesh files against a design specification using MeshLib. Reports findings only — does not make pass/fail decisions.",
    instruction=INSTRUCTION,
    tools=[execute_meshlib_code, explore_meshlib_api],
)


def run_inspection(mesh_path: str, design_brief: dict, output_dir: str, outer_attempt: int = 1) -> dict:
    """Run the MeshLib inspection agent and return structured findings.
    
    Args:
        mesh_path: Path to the STL file.
        design_brief: The design specification dict.
        output_dir: The timestamped directory for saving outputs.
        outer_attempt: The current iteration of the outer redesign loop.
    
    Returns:
        A dict with keys: checks, anomalies, engineer_summary, confidence.
    """
    global _current_output_dir, _current_outer_attempt, _script_counter
    _current_output_dir = output_dir
    _current_outer_attempt = outer_attempt
    _script_counter = 0  # Reset per outer iteration for clean numbering
    
    # 1. Run invariant baseline
    baseline = run_invariant_baseline(mesh_path)
    if baseline.get("load_failed"):
        return {
            "checks": [],
            "anomalies": [f"Mesh loading failed: {baseline.get('hard_failures', ['Unknown'])[0]}"],
            "engineer_summary": "The mesh file could not be loaded for inspection.",
            "confidence": "HIGH"
        }

    # 2. Build the message for the agent
    message = (
        f"Mesh Path: {mesh_path}\n\n"
        f"Design Brief:\n{json.dumps(design_brief, indent=2)}\n\n"
        f"Baseline Results:\n{json.dumps(baseline, indent=2)}\n\n"
        f"Instruction: The baseline has already checked watertightness, volume, "
        f"self-intersections, and bounding box — do not repeat these. "
        f"Focus on plan-specific dimensional and feature verification. "
        f"Report ALL findings with exact measurements."
    )

    # 3. Set up session
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

    # 5. Run
    content = types.Content(role='user', parts=[types.Part(text=message)])
    events = []
    try:
        events = list(runner.run(user_id="user", session_id=session_id, new_message=content))
    except Exception as e:
        logging.getLogger("google_adk").error(f"MeshLib agent error: {e}")
        return {
            "checks": [],
            "anomalies": [f"Agent execution failed: {e}"],
            "engineer_summary": f"The MeshLib inspection agent failed: {e}",
            "confidence": "LOW"
        }

    # 6. Extract final response
    final_text = None
    for event in events:
        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = event.content.parts[0].text
                break

    # 7. Parse JSON
    if final_text:
        cleaned = final_text.replace('```json', '').replace('```', '').strip()
        try:
            findings = json.loads(cleaned)
            for key in ["checks", "anomalies", "engineer_summary", "confidence"]:
                if key not in findings:
                    findings[key] = [] if key in ("checks", "anomalies") else None
            
            # Save conversation for debugging
            try:
                os.makedirs(output_dir, exist_ok=True)
                convo_path = os.path.join(output_dir, f"06b_outer{outer_attempt}_ai_inspector_conversation_trace.json")
                with open(convo_path, "w") as f:
                    json.dump([e.model_dump(mode='json') for e in events], f, indent=4)
            except Exception:
                pass
            
            return findings
        except json.JSONDecodeError:
            pass

    # 8. Fallback
    return {
        "checks": [],
        "anomalies": [f"Could not parse agent response: {final_text}"],
        "engineer_summary": final_text or "No response received from agent.",
        "confidence": "LOW"
    }
