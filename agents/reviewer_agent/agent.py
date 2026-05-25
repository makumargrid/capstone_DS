"""
Adversarial Reviewer Agent — Pure reasoning, no tools, information asymmetry.

This agent receives:
  1. The Design Brief (what was requested)
  2. Static check results (deterministic ground truth from mesh_inspector.py)
  3. MeshLib AI inspection findings (LLM-generated measurements)

It does NOT see the generated CadQuery code (information asymmetry).
Its job is to cross-reference findings, catch hallucinated measurements,
and make a routing decision: APPROVED / REDESIGN / HALT.
"""

import json
import os
import logging
import asyncio
import uuid

os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'false'

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types


INSTRUCTION = """You are a Senior Quality Assurance Reviewer for a CAD manufacturing pipeline.

Your role is adversarial: you must CHALLENGE the findings presented to you, not blindly trust them.

You will receive three inputs:
1. **Design Brief** — The original engineering specification (what was requested).
2. **Static Check Results** — Deterministic, code-based measurements (bounding box, watertightness, volume, self-intersections, wall thickness). These are GROUND TRUTH — they come from hardcoded math, not an LLM.
3. **AI Inspection Findings** — Measurements and observations from an AI agent that wrote and executed MeshLib code at runtime. These CAN be wrong if the AI wrote buggy code.

Your workflow:
1. Read the Design Brief to understand the engineer's intent.
2. Read the Static Check Results as your baseline truth.
3. Read the AI Inspection Findings and cross-reference every measurement against the Static Check Results.
4. If the AI findings contradict the static results (e.g., AI says bounding box is 500mm but static says 130mm), flag this as a discrepancy and TRUST THE STATIC RESULTS.
5. If both static and AI agree on a failure (e.g., both show wall thickness below spec), this is a genuine design problem.
6. Make your decision.

Decision rules:
- **APPROVED**: All static checks passed AND the AI findings show no genuine design problems. Minor discrepancies that are within tolerance are acceptable.
- **REDESIGN**: There is a genuine, confirmed design problem that requires the Planner Agent to regenerate the CAD code. You MUST provide specific, actionable recommendations (e.g., "Increase blade thickness from 2mm to 3mm at the tip").
- **HALT**: The results are contradictory, uninterpretable, or indicate a systemic issue (e.g., mesh won't load, segfault during inspection). Human review is required.

IMPORTANT: You have NO tools. You cannot execute code, read files, or call APIs.
You are a pure reasoning agent. Your power comes from cross-referencing data sources.

Output ONLY a JSON object with these exact keys (no markdown wrapping):
{
  "decision": "APPROVED" | "REDESIGN" | "HALT",
  "reasoning": "Detailed chain of thought explaining your decision step by step",
  "discrepancies_found": ["List of contradictions between static and AI results, if any"],
  "recommendations_for_planner": "Specific actionable changes if REDESIGN, else null",
  "confidence": "HIGH" | "MEDIUM" | "LOW"
}
"""

root_agent = Agent(
    name="adversarial_reviewer",
    model="gemini-3.1-pro-preview",
    description="Adversarial quality reviewer that cross-references deterministic static checks against AI inspection findings to catch hallucinations and make routing decisions.",
    instruction=INSTRUCTION,
    tools=[],  # NO TOOLS — pure reasoning agent
)


def run_adversarial_review(
    design_brief: dict,
    static_results: dict,
    ai_findings: dict,
) -> dict:
    """
    Run the adversarial review process.
    
    Args:
        design_brief: The original engineering specification.
        static_results: Deterministic results from mesh_inspector.py.
        ai_findings: LLM-generated findings from the meshlib agent.
    
    Returns:
        A dict with keys: decision, reasoning, discrepancies_found,
        recommendations_for_planner, confidence.
    """
    # Build the message for the reviewer
    message = (
        f"=== DESIGN BRIEF ===\n{json.dumps(design_brief, indent=2)}\n\n"
        f"=== STATIC CHECK RESULTS (GROUND TRUTH) ===\n{json.dumps(static_results, indent=2)}\n\n"
        f"=== AI INSPECTION FINDINGS ===\n{json.dumps(ai_findings, indent=2)}\n\n"
        f"Cross-reference the above and produce your verdict."
    )

    # Set up session
    session_id = str(uuid.uuid4())
    db_dir = "outputs"
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "adk_sessions.db")
    db_url = f"sqlite:///{os.path.abspath(db_path)}"
    session_service = DatabaseSessionService(db_url=db_url)

    async def create_session():
        await session_service.create_session(
            app_name="reviewer_agent",
            user_id="user",
            session_id=session_id
        )
    asyncio.run(create_session())

    runner = Runner(
        agent=root_agent,
        app_name="reviewer_agent",
        session_service=session_service
    )

    content = types.Content(role='user', parts=[types.Part(text=message)])

    # Run the agent
    events = []
    try:
        events = list(runner.run(user_id="user", session_id=session_id, new_message=content))
    except Exception as e:
        logging.getLogger("google_adk").error(f"Reviewer agent error: {e}")
        return {
            "decision": "HALT",
            "reasoning": f"Reviewer agent failed to execute: {e}",
            "discrepancies_found": [],
            "recommendations_for_planner": None,
            "confidence": "LOW"
        }

    # Extract final response
    final_text = None
    for event in events:
        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = event.content.parts[0].text
                break

    if final_text:
        cleaned = final_text.replace('```json', '').replace('```', '').strip()
        try:
            verdict = json.loads(cleaned)
            # Ensure required keys
            for key in ["decision", "reasoning", "discrepancies_found", "recommendations_for_planner", "confidence"]:
                if key not in verdict:
                    verdict[key] = None
            return verdict
        except json.JSONDecodeError:
            pass

    # Fallback if we couldn't parse
    return {
        "decision": "HALT",
        "reasoning": f"Could not parse reviewer response: {final_text}",
        "discrepancies_found": [],
        "recommendations_for_planner": None,
        "confidence": "LOW"
    }
