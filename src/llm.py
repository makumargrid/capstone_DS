"""
LLM Engine — Planner Agent with plan-before-code workflow.

The Planner Agent operates in two phases within a single persistent session:
  Phase 1: Design Planning — thinks like a real CAD engineer, decomposes the part
  Phase 2: Code Generation — writes CadQuery code based on its own plan

The session persists across outer-loop iterations so the agent accumulates context
about previous failures and reviewer feedback.
"""

import os
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'false'

import sys
import time
import json
import uuid
import asyncio
from google import genai
from dotenv import load_dotenv

from src.logger import get_agent_logger

load_dotenv()
logger = get_agent_logger()


# ---------------------------------------------------------------------------
# ask_user tool — allows the planner to request clarification
# ---------------------------------------------------------------------------

def ask_user(question: str) -> str:
    """Ask the user a clarifying question about the design requirements.
    
    Use this ONLY when the prompt is genuinely ambiguous about critical
    dimensions, tolerances, material properties, or manufacturing constraints.
    Do NOT ask trivial questions — make your best engineering judgment for
    minor details.
    
    Args:
        question: The specific question to ask the user.
    
    Returns:
        The user's answer as a string.
    """
    # Check if we're running in interactive mode (stdin is a TTY)
    if sys.stdin.isatty():
        print(f"\n🤔 PLANNER QUESTION: {question}")
        print(">>> ", end="", flush=True)
        answer = input()
        logger.info(f"User answered: {answer}")
        return answer
    else:
        # Non-interactive mode (Docker without -it) — tell the agent to proceed
        logger.info(f"Non-interactive mode. Planner asked: {question}. Proceeding with best judgment.")
        return (
            "You are running in non-interactive mode. The user cannot answer right now. "
            "Please proceed with your best engineering judgment based on the available information."
        )


# ---------------------------------------------------------------------------
# Planner Agent system prompt
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """You are a Senior CAD Engineer with 20 years of experience in parametric modeling.
You use CadQuery (Python) to build 3D solid models.

## CANONICAL COORDINATE SYSTEM (all agents use this — never deviate)
- **Axis**: Z-up. The part sits on the XY plane at Z=0 and extends upward along +Z.
- **Units**: ALL dimensions are in millimeters (mm). No exceptions.
- **Origin**: The center of the part base is at (0, 0, 0).
- When receiving feedback, all measurements reference this exact coordinate system.
  e.g., "face at Z=30mm has wall thickness 0.8mm" means exactly what it says.

## DFM (Design for Manufacturing) CONSTRAINTS
Your designs must be physically manufacturable. Always ensure:
- **Minimum wall thickness**: 2.0mm everywhere (no exceptions, even at blade tips or edges).
- **No unsupported overhangs** > 45° for FDM printing (unless explicitly allowed).
- **Watertight manifold**: The final solid must be a single closed manifold with no holes.
- These are NOT optional — a part that fails DFM is a failed part.

YOUR WORKFLOW (follow this order strictly):

## Phase 1: CLARIFY (if needed)
If the user's prompt is ambiguous about critical dimensions, tolerances, or
manufacturing constraints, use the `ask_user` tool to clarify. Examples:
- "The prompt says 'a few holes' — how many exactly?"
- "No wall thickness specified — should I use 2mm minimum?"
Do NOT ask trivial questions. Use your engineering judgment for minor details.

## Phase 2: PLAN (mandatory — always do this)
Before writing ANY code, produce a Construction Plan. Think like a real engineer:

1. **Decompose** the part into sub-components (e.g., Hub, Bore, Blade_1..Blade_N).
2. For each sub-component, specify:
   - The CadQuery operation (workplane, extrude, revolve, loft, sweep, cut, fuse)
   - Exact dimensions in mm, referenced to the canonical Z-up axis
   - How it attaches to other components (boolean union, cut, etc.)
3. **Assembly order**: which parts build on which (e.g., "Start with hub, then cut bore, then fuse blades").
4. **Risk assessment**: identify geometries that CadQuery may struggle with (complex sweeps, thin features, self-intersecting booleans).

Output your plan as structured text under the heading "CONSTRUCTION PLAN:".

## Phase 3: CODE (mandatory — always do this after the plan)
Write the complete CadQuery Python code based on your plan.

Code rules:
- Output ONLY valid Python code after your plan. No markdown fencing.
- Import cadquery as cq. Import math if needed.
- Assign the final solid to `result_solid`.
- Use standard CadQuery syntax and selectors (>Z, <Z, >X, not >Z, %Plane).
- NEVER hallucinate or invent custom selectors (no LargestAreaSelector etc.).
- Ensure all dimensions are in millimeters.
- The code must produce a single unified solid (use .union() / .cut() as needed).

## CADQUERY ENGINEERING TIPS (critical for twisted/swept geometry)

### Wall Thickness in Twisted Lofts
When you loft rectangular profiles along a twisting path (e.g., impeller blades):
- The HORIZONTAL profile width ≠ the NORMAL (true 3D) wall thickness.
- Due to the twist, the normal thickness is LESS than the horizontal width by a factor of cos(pitch_angle).
- To guarantee a target normal thickness T, set horizontal half-width to:
  `half_width = (T / 2.0) / cos(atan(radius * twist_rate_radians_per_mm))`
- **CRITICAL**: If a static check reports wall thickness X mm (where X < target),
  DO NOT just tweak the formula — instead multiply your current profile width by (target / X) * 1.2
  (the 1.2 gives a 20% safety margin to account for lofting interpolation artifacts).
- Example: If target is 2.0mm and static check measured 1.0mm, double your profile width + 20%.

### Boolean Union Tips
- Always embed sub-components 1-2mm into the parent body to ensure clean boolean unions.
- Avoid coplanar faces at Z=0 or Z=max — offset cutting tools by a small epsilon (e.g., -1mm).

When you receive FEEDBACK from a previous failed attempt:
- Read the EXACT NUMBERS carefully (measured thickness, deficit, number of thin regions).
- Identify which section of your code controls the failing dimension.
- Apply the quantitative fix described in the feedback (e.g., "multiply width by 2.4").
- Explain what you changed and why.
"""


# ---------------------------------------------------------------------------
# Helper: LLM call with fallback
# ---------------------------------------------------------------------------

def _call_llm_with_fallback(client, model_name: str, contents: str) -> str:
    """Helper to call generate_content with retries and fallback to gemini-2.5-flash."""
    models_to_try = [model_name]
    if model_name != 'gemini-2.5-flash':
        models_to_try.append('gemini-2.5-flash')
        
    last_error = None
    for model in models_to_try:
        for attempt in range(3):
            try:
                logger.info(f"Generating content using {model} (attempt {attempt+1}/3)...")
                response = client.models.generate_content(
                    model=model,
                    contents=contents
                )
                return response.text
            except Exception as e:
                last_error = e
                logger.warning(f"Error generating content with {model} on attempt {attempt+1}: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        logger.error(f"All attempts failed with model {model}. Trying fallback/next model if available.")
    
    raise last_error


# ---------------------------------------------------------------------------
# Planner Agent — persistent session, plan+code workflow
# ---------------------------------------------------------------------------

class PlannerAgent:
    """Manages a persistent ADK agent session for the planner.
    
    The session persists across outer-loop iterations so the agent accumulates
    context about previous failures and reviewer feedback.
    """
    
    def __init__(self, model_name: str = 'gemini-3.1-pro-preview', interactive: bool = False):
        from google.adk.agents.llm_agent import Agent
        from google.adk.runners import Runner
        from google.adk.sessions import DatabaseSessionService
        
        self.model_name = model_name
        self.interactive = interactive
        
        # Build tools list
        tools = [ask_user] if interactive else []
        
        self.agent = Agent(
            model=model_name,
            name='planner_agent',
            description='Senior CAD engineer that plans before coding.',
            instruction=PLANNER_SYSTEM_PROMPT,
            tools=tools,
        )
        
        # Set up persistent session
        db_dir = "outputs"
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "adk_sessions.db")
        db_url = f"sqlite:///{os.path.abspath(db_path)}"
        
        self.session_service = DatabaseSessionService(db_url=db_url)
        self.session_id = str(uuid.uuid4())
        
        async def create_session():
            await self.session_service.create_session(
                app_name="planner_agent",
                user_id="user",
                session_id=self.session_id
            )
        asyncio.run(create_session())
        
        self.runner = Runner(
            agent=self.agent,
            app_name="planner_agent",
            session_service=self.session_service
        )
        
        logger.info(f"PlannerAgent initialized with model={model_name}, session={self.session_id}")
    
    def generate(self, message: str) -> str:
        """Send a message to the planner and get the full response.
        
        Args:
            message: The user message or feedback message.
        
        Returns:
            The full text response from the agent.
        """
        from google.genai import types
        
        content = types.Content(role='user', parts=[types.Part(text=message)])
        events = list(self.runner.run(
            user_id="user",
            session_id=self.session_id,
            new_message=content
        ))
        
        response_text = ""
        for event in events:
            if event.is_final_response() and event.content and event.content.parts:
                response_text = event.content.parts[0].text
                break
        
        if not response_text:
            raise RuntimeError("No response from planner_agent.")
        
        return response_text
    
    def generate_cad_code(self, prompt: str) -> tuple:
        """Generate a Construction Plan + CadQuery code from a prompt.
        
        Returns:
            (full_response, extracted_code) tuple.
        """
        logger.info("Requesting Construction Plan + Code from planner_agent...")
        
        full_response = self.generate(f"Request: {prompt}")
        code = self._extract_code(full_response)
        
        return full_response, code
    
    def regenerate_with_feedback(self, feedback: str) -> tuple:
        """Send failure feedback and get regenerated code.
        
        The session is persistent, so the agent has full context of previous attempts.
        
        Args:
            feedback: Detailed failure description and recommendations.
        
        Returns:
            (full_response, extracted_code) tuple.
        """
        logger.info("Sending feedback to planner_agent for regeneration...")
        
        message = (
            f"FEEDBACK FROM QUALITY REVIEW:\n\n"
            f"{feedback}\n\n"
            f"Please update your Construction Plan to address these issues, "
            f"then regenerate the complete CadQuery code."
        )
        
        full_response = self.generate(message)
        code = self._extract_code(full_response)
        
        return full_response, code
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from the agent's response.
        
        The agent outputs a plan followed by code. We extract just the code portion.
        """
        # Try to find code between ```python markers
        if '```python' in response:
            parts = response.split('```python')
            if len(parts) > 1:
                code_block = parts[-1].split('```')[0]
                return code_block.strip()
        
        # Try to find code between ``` markers
        if '```' in response:
            parts = response.split('```')
            # Find the largest code block (likely the CadQuery code)
            code_blocks = []
            for i in range(1, len(parts), 2):
                block = parts[i].strip()
                # Remove language identifier if present
                if block.startswith(('python', 'py')):
                    block = block[block.index('\n')+1:]
                code_blocks.append(block)
            
            if code_blocks:
                # Return the longest code block (most likely the full CadQuery script)
                return max(code_blocks, key=len).strip()
        
        # Fallback: look for 'import cadquery' as a marker
        lines = response.split('\n')
        code_start = None
        for i, line in enumerate(lines):
            if 'import cadquery' in line or 'import cq' in line:
                code_start = i
                break
        
        if code_start is not None:
            return '\n'.join(lines[code_start:]).strip()
        
        # Last resort: strip any markdown and return the whole thing
        cleaned = response.replace('```python', '').replace('```', '').strip()
        return cleaned


# ---------------------------------------------------------------------------
# Standalone dimension extraction (used before the main loop)
# ---------------------------------------------------------------------------

def extract_expected_dimensions(prompt: str, model_name: str = 'gemini-3.1-pro-preview') -> dict:
    """Uses the LLM to dynamically predict the expected bounding box of the object based on the prompt."""
    logger.info("Dynamically calculating expected dimensions from prompt...")
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    system_prompt = """
You are a geometry analysis agent. Based on the user's CAD prompt, calculate the expected bounding box dimensions (X, Y, Z in millimeters).
Return ONLY a valid JSON object in this exact format, with no markdown formatting:
{
    "x_mm": float,
    "y_mm": float,
    "z_mm": float,
    "tolerance_mm": float
}
Set the tolerance based on how complex/curved the object is (e.g., 1.0 for simple boxes, 5.0-10.0 for complex radial/curved objects).
"""
    full_prompt = f"{system_prompt}\nRequest:{prompt}"
    
    try:
        response_text = _call_llm_with_fallback(client, model_name, full_prompt)
        clean_text = response_text.replace('```json', '').replace('```', '').strip()
        dims = json.loads(clean_text)
        logger.info(f"Dynamically calculated dimensions: {dims}")
        return dims
    except Exception as e:
        logger.error(f"Failed to extract dimensions dynamically: {e}")
        return {"x_mm": 100.0, "y_mm": 100.0, "z_mm": 100.0, "tolerance_mm": 10.0}
