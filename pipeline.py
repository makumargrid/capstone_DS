"""
Adversarial Multi-Agent CAD Pipeline

Architecture:
  Phase 1: Planning + Code Generation (Planner Agent with ask_user)
  Phase 2: CadQuery Execution (inner retry for syntax errors)
  Phase 3: Export STL + STEP
  Phase 4: Static Checks (deterministic — mesh_inspector.py)
  Phase 5: AI Inspection (MeshLib Agent — findings only, no verdict)
  Phase 6: Adversarial Review (Reviewer Agent — cross-references static + AI)

The outer loop (Phases 2→6) feeds reviewer recommendations back to the
Planner Agent's persistent session, enabling iterative refinement.
"""

import os
import sys
import datetime
import json

from src.logger import get_agent_logger
from src.llm import PlannerAgent, extract_expected_dimensions
from src.cad_executor import execute_cad_code, export_solid
from src.mesh_inspector import run_all_inspections, has_hard_failures, has_soft_failures
from agents.meshlib_agent import run_inspection
from agents.reviewer_agent import run_adversarial_review


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CODE_RETRIES = 3     # Inner loop: fix syntax/execution errors
MAX_OUTER_RETRIES = 3    # Outer loop: fix design intent failures


# ---------------------------------------------------------------------------
# Helper: Smart code excerpt extraction
# ---------------------------------------------------------------------------

def _extract_critical_code_section(full_code: str, keyword: str = "blade") -> str:
    """Extract only the critical section of generated code relevant to the failure.
    
    Instead of sending the full 100-line code (which wastes context), we extract
    only the section most likely responsible for the problem (e.g., the blade
    generation loop).
    
    Strategy:
    1. Find the first line containing the keyword (case-insensitive).
    2. Walk backward to find the section start (a comment line starting with #).
    3. Walk forward to find the section end (the next blank line followed by a comment,
       or a line starting with a new section marker like "# ---").
    4. Return that section (capped at 40 lines to stay context-efficient).
    """
    lines = full_code.split('\n')
    keyword_lower = keyword.lower()
    
    # Find the first line containing the keyword
    start_idx = None
    for i, line in enumerate(lines):
        if keyword_lower in line.lower():
            start_idx = i
            break
    
    if start_idx is None:
        # Keyword not found — return the middle chunk of the code
        mid = len(lines) // 2
        return '\n'.join(lines[max(0, mid - 20):mid + 20])
    
    # Walk backward to find the section header
    section_start = start_idx
    for i in range(start_idx - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith('# ---') or stripped.startswith('# ==='):
            section_start = i
            break
        if stripped == '' and i < start_idx - 1:
            section_start = i + 1
            break
    
    # Walk forward to find the section end
    section_end = min(start_idx + 30, len(lines))
    for i in range(start_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith('# ---') or stripped.startswith('# ==='):
            section_end = i
            break
    
    # Cap at 40 lines
    excerpt = '\n'.join(lines[section_start:min(section_end, section_start + 40)])
    return excerpt


def run_pipeline(request_prompt: str, output_base_dir: str = "outputs", interactive: bool = False):
    """
    Run the full adversarial multi-agent CAD pipeline.
    
    Args:
        request_prompt: The natural language CAD design prompt.
        output_base_dir: Base directory for all outputs.
        interactive: If True, the planner agent can ask the user questions via stdin.
    """
    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "00_pipeline_execution.log")
    logger = get_agent_logger(log_file)
    
    logger.info("=" * 70)
    logger.info("ADVERSARIAL MULTI-AGENT CAD PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Interactive mode: {interactive}")
    logger.info(f"Prompt: {request_prompt[:200]}...")
    
    # -----------------------------------------------------------------------
    # Pre-computation: Extract expected dimensions
    # -----------------------------------------------------------------------
    logger.info("[PRE] Extracting expected dimensions from prompt...")
    expected_dimensions = extract_expected_dimensions(request_prompt)
    logger.info(f"[PRE] Expected dimensions: {expected_dimensions}")
    
    # Build Design Brief — this is the structured specification that all agents reference
    design_brief = {
        "original_prompt": request_prompt,
        "expected_dims": expected_dimensions,
        "min_wall_mm": 2.0,
        "manufacturing_process": None,
        "primitives": []
    }
    
    # Save the design brief
    brief_path = os.path.join(output_dir, "01_design_brief.json")
    with open(brief_path, "w") as f:
        json.dump(design_brief, f, indent=4)
    logger.info(f"[PRE] Design brief saved to {brief_path}")
    
    # -----------------------------------------------------------------------
    # Initialize Planner Agent (persistent session across all iterations)
    # -----------------------------------------------------------------------
    logger.info("[INIT] Initializing Planner Agent with persistent session...")
    planner = PlannerAgent(interactive=interactive)
    
    # -----------------------------------------------------------------------
    # Phase 1: Initial Planning + Code Generation
    # -----------------------------------------------------------------------
    logger.info("[PHASE 1] Requesting Construction Plan + Code from Planner Agent...")
    try:
        full_response, code = planner.generate_cad_code(request_prompt)
    except Exception as e:
        logger.error(f"[PHASE 1] Planner Agent failed: {e}")
        return
    
    # Save the plan + code response (this is the initial attempt before entering the loop)
    # We'll save it as outer iteration 0 just to capture the initial state.
    plan_path = os.path.join(output_dir, "02_outer0_planner_construction_plan.txt")
    with open(plan_path, "w") as f:
        f.write(full_response)
    logger.info(f"[PHASE 1] Initial construction plan saved to {plan_path}")
    
    # -----------------------------------------------------------------------
    # Outer Loop: Generate → Execute → Inspect → Review → (Feedback)
    # -----------------------------------------------------------------------
    for outer_attempt in range(1, MAX_OUTER_RETRIES + 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"OUTER LOOP — Iteration {outer_attempt}/{MAX_OUTER_RETRIES}")
        logger.info(f"{'='*70}")
        
        # -------------------------------------------------------------------
        # Phase 2: Execute code (with inner retry for syntax errors)
        # -------------------------------------------------------------------
        solid = None
        current_code = code
        
        for code_attempt in range(1, MAX_CODE_RETRIES + 1):
            logger.info(f"[PHASE 2] Code execution attempt {code_attempt}/{MAX_CODE_RETRIES}...")
            
            # Save the code
            code_path = os.path.join(output_dir, f"03_outer{outer_attempt}_inner{code_attempt}_planner_generated_cad_code.py")
            with open(code_path, "w") as f:
                f.write(current_code)
            logger.info(f"[PHASE 2] Code saved to {code_path}")
            
            try:
                solid = execute_cad_code(current_code)
                logger.info("[PHASE 2] ✅ Solid generated successfully!")
                break
            except Exception as e:
                logger.error(f"[PHASE 2] ❌ Code execution failed: {e}")
                if code_attempt < MAX_CODE_RETRIES:
                    logger.info("[PHASE 2] Sending error feedback to Planner for code fix...")
                    feedback = (
                        f"CODE EXECUTION ERROR (attempt {code_attempt}):\n"
                        f"```\n{str(e)}\n```\n"
                        f"Fix the Python/CadQuery code to resolve this error. "
                        f"Output the complete corrected code."
                    )
                    try:
                        _, current_code = planner.regenerate_with_feedback(feedback)
                    except Exception as regen_err:
                        logger.error(f"[PHASE 2] Regeneration failed: {regen_err}")
                        break
                else:
                    logger.error(f"[PHASE 2] Max code retries ({MAX_CODE_RETRIES}) exhausted.")
        
        if solid is None:
            logger.error("[PHASE 2] Could not generate a valid solid. Pipeline failed.")
            return
        
        # -------------------------------------------------------------------
        # Phase 3: Export STEP + STL
        # -------------------------------------------------------------------
        logger.info("[PHASE 3] Exporting solid to STEP and STL...")
        step_path = os.path.join(output_dir, f"04_outer{outer_attempt}_exported_model.step")
        stl_path = os.path.join(output_dir, f"04_outer{outer_attempt}_exported_model.stl")
        
        try:
            export_solid(solid, step_path)
            export_solid(solid, stl_path)
            logger.info(f"[PHASE 3] ✅ Exported STEP to {step_path}")
            logger.info(f"[PHASE 3] ✅ Exported STL to {stl_path}")
        except Exception as e:
            logger.error(f"[PHASE 3] Export failed: {e}")
            return
        
        # -------------------------------------------------------------------
        # Phase 4: Static Checks (deterministic — no LLM)
        # -------------------------------------------------------------------
        logger.info("[PHASE 4] Running deterministic static checks...")
        # Note: mesh_inspector saves 05_outerX_static_inspection_ground_truth.json internally
        static_results = run_all_inspections(stl_path, expected_dimensions, output_dir, outer_attempt)
        
        if has_hard_failures(static_results):
            failures = static_results.get("hard_failures", [])
            logger.warning(f"[PHASE 4] ❌ Static checks found {len(failures)} hard failure(s):")
            for f in failures:
                logger.warning(f"  → {f}")
            
            if outer_attempt < MAX_OUTER_RETRIES:
                # Short-circuit: route back to planner WITHOUT invoking AI agents
                logger.info("[PHASE 4] Short-circuiting to Planner (skipping AI inspection)...")
                feedback = (
                    f"STATIC CHECK FAILURES (deterministic, ground truth):\n"
                    + "\n".join(f"- {f}" for f in failures)
                    + "\n\nThese are mathematically verified failures, not LLM opinions. "
                    + "Please redesign the part to fix these issues. "
                    + "Update your Construction Plan and regenerate the code."
                )
                try:
                    full_response, code = planner.regenerate_with_feedback(feedback)
                    # Save updated plan
                    plan_path = os.path.join(output_dir, f"02_outer{outer_attempt}_planner_construction_plan.txt")
                    with open(plan_path, "w") as f_plan:
                        f_plan.write(full_response)
                except Exception as e:
                    logger.error(f"[PHASE 4] Planner regeneration failed: {e}")
                    return
                continue  # Next outer iteration
            else:
                logger.error("[PHASE 4] Max outer retries exhausted with static failures.")
                return
        
        logger.info("[PHASE 4] ✅ No hard failures (geometry is valid).")
        
        # -------------------------------------------------------------------
        # Phase 4b: DFM Soft Failure Check
        # -------------------------------------------------------------------
        if has_soft_failures(static_results):
            soft_fails = static_results.get("soft_failures", [])
            logger.warning(f"[PHASE 4b] ⚠ DFM soft failure(s) detected ({len(soft_fails)}):")
            for sf in soft_fails:
                logger.warning(f"  → {sf}")
            
            if outer_attempt < MAX_OUTER_RETRIES:
                logger.info("[PHASE 4b] Short-circuiting to Planner (skipping AI — we already know the answer)...")
                
                # Build enriched quantitative feedback
                wt = static_results.get("wall_thickness", {})
                code_excerpt = _extract_critical_code_section(code, "blade")
                
                feedback = (
                    f"DFM STATIC CHECK FAILURE — Iteration {outer_attempt}/{MAX_OUTER_RETRIES}\n"
                    f"{'='*60}\n\n"
                    f"QUANTITATIVE DATA (these are mathematically exact, not AI opinions):\n"
                    f"  • Measured minimum wall thickness: {wt.get('min_wall_thickness_mm', 'N/A')}mm\n"
                    f"  • Required minimum wall thickness: 2.0mm\n"
                    f"  • Deficit: {round(2.0 - (wt.get('min_wall_thickness_mm') or 0), 3)}mm\n"
                    f"  • Number of thin regions: {wt.get('thin_region_count', 0)}\n"
                    f"  • Thin region samples: {json.dumps(wt.get('thin_regions_sample', []))}\n\n"
                    f"ACTION REQUIRED:\n"
                    f"  Your blade profile half-width is too small.\n"
                    f"  Multiply your current profile width by: "
                    f"{round(2.0 / max(wt.get('min_wall_thickness_mm', 1.0), 0.1) * 1.2, 2)}x\n"
                    f"  (This is target/measured * 1.2 safety margin)\n\n"
                    f"RELEVANT CODE SECTION TO FIX:\n```python\n{code_excerpt}\n```\n\n"
                    f"Coordinate system: Z-up, all dimensions in mm, origin at (0,0,0).\n"
                    f"Update your Construction Plan and regenerate the COMPLETE code."
                )
                try:
                    full_response, code = planner.regenerate_with_feedback(feedback)
                    plan_path = os.path.join(output_dir, f"02_outer{outer_attempt}_planner_construction_plan.txt")
                    with open(plan_path, "w") as f_plan:
                        f_plan.write(full_response)
                except Exception as e:
                    logger.error(f"[PHASE 4b] Planner regeneration failed: {e}")
                    return
                continue  # Next outer iteration
            else:
                logger.error("[PHASE 4b] Max outer retries exhausted with DFM soft failures.")
                return
        
        logger.info("[PHASE 4] ✅ All static AND DFM checks passed!")
        
        # -------------------------------------------------------------------
        # Phase 5: AI Inspection (MeshLib Agent — findings only)
        # -------------------------------------------------------------------
        logger.info("[PHASE 5] Running MeshLib AI Inspector Agent...")
        ai_findings = run_inspection(stl_path, design_brief, output_dir, outer_attempt)
        
        # Save AI findings
        ai_findings_path = os.path.join(output_dir, f"06a_outer{outer_attempt}_ai_inspector_findings.json")
        with open(ai_findings_path, "w") as f:
            json.dump(ai_findings, f, indent=4)
        logger.info(f"[PHASE 5] AI findings saved to {ai_findings_path}")
        logger.info(f"[PHASE 5] AI Summary: {ai_findings.get('engineer_summary', 'N/A')}")
        logger.info(f"[PHASE 5] AI Confidence: {ai_findings.get('confidence', 'N/A')}")
        
        # -------------------------------------------------------------------
        # Phase 6: Adversarial Review
        # -------------------------------------------------------------------
        logger.info("[PHASE 6] Running Adversarial Reviewer Agent...")
        logger.info("[PHASE 6] (Reviewer sees: Design Brief + Static Results + AI Findings)")
        logger.info("[PHASE 6] (Reviewer does NOT see: generated CadQuery code)")
        
        reviewer_verdict = run_adversarial_review(
            design_brief=design_brief,
            static_results=static_results,
            ai_findings=ai_findings,
        )
        
        # Save reviewer verdict
        verdict_path = os.path.join(output_dir, f"07_outer{outer_attempt}_adversarial_reviewer_verdict.json")
        with open(verdict_path, "w") as f:
            json.dump(reviewer_verdict, f, indent=4)
        
        decision = reviewer_verdict.get("decision", "HALT")
        confidence = reviewer_verdict.get("confidence", "LOW")
        reasoning = reviewer_verdict.get("reasoning", "No reasoning provided")
        
        logger.info(f"[PHASE 6] Decision: {decision} (Confidence: {confidence})")
        logger.info(f"[PHASE 6] Reasoning: {reasoning[:300]}...")
        
        if reviewer_verdict.get("discrepancies_found"):
            logger.info("[PHASE 6] Discrepancies found between static and AI results:")
            for d in reviewer_verdict["discrepancies_found"]:
                logger.info(f"  ⚠ {d}")
        
        # -------------------------------------------------------------------
        # Route based on reviewer decision
        # -------------------------------------------------------------------
        if decision == "APPROVED":
            logger.info("=" * 70)
            logger.info("✅ PIPELINE COMPLETE — Design APPROVED by adversarial review!")
            logger.info("=" * 70)
            logger.info(f"Final outputs saved in: {output_dir}")
            return
        
        elif decision == "REDESIGN":
            recommendations = reviewer_verdict.get("recommendations_for_planner", "No specific recommendations.")
            logger.warning(f"[PHASE 6] 🔄 REDESIGN requested. Recommendations: {recommendations}")
            
            if outer_attempt < MAX_OUTER_RETRIES:
                logger.info(f"[PHASE 6] Feeding reviewer recommendations back to Planner (iteration {outer_attempt+1})...")
                
                # Enriched feedback with quantitative data + smart code excerpt
                wt = static_results.get("wall_thickness", {})
                code_excerpt = _extract_critical_code_section(code, "blade")
                
                feedback = (
                    f"ADVERSARIAL REVIEW — REDESIGN REQUIRED (Iteration {outer_attempt}/{MAX_OUTER_RETRIES})\n"
                    f"{'='*60}\n\n"
                    f"REVIEWER REASONING:\n{reasoning}\n\n"
                    f"SPECIFIC RECOMMENDATIONS:\n{recommendations}\n\n"
                    f"QUANTITATIVE STATIC CHECK DATA:\n"
                    f"  • Wall thickness: min={wt.get('min_wall_thickness_mm', 'N/A')}mm, "
                    f"thin_regions={wt.get('thin_region_count', 0)}\n"
                    f"  • Bounding box: {json.dumps(static_results.get('dimensions_check', {}).get('dimensions', {}), indent=2)}\n\n"
                    f"RELEVANT CODE SECTION TO FIX:\n```python\n{code_excerpt}\n```\n\n"
                    f"Coordinate system: Z-up, all dimensions in mm, origin at (0,0,0).\n"
                    f"Update your Construction Plan and regenerate the COMPLETE code."
                )
                try:
                    full_response, code = planner.regenerate_with_feedback(feedback)
                    plan_path = os.path.join(output_dir, f"02_outer{outer_attempt}_planner_construction_plan_updated.txt")
                    with open(plan_path, "w") as f_plan:
                        f_plan.write(full_response)
                except Exception as e:
                    logger.error(f"[PHASE 6] Planner regeneration failed: {e}")
                    return
                continue  # Next outer iteration
            else:
                logger.error("[PHASE 6] Max outer retries exhausted. Design not approved.")
                return
        
        elif decision == "HALT":
            logger.error("=" * 70)
            logger.error("🛑 PIPELINE HALTED — Human review required!")
            logger.error("=" * 70)
            logger.error(f"Reason: {reasoning}")
            logger.error(f"Review the outputs in: {output_dir}")
            return
        
        else:
            logger.error(f"[PHASE 6] Unknown decision: {decision}. Halting.")
            return
    
    # If we exhausted all outer iterations without APPROVED
    logger.warning("Pipeline completed all outer iterations without achieving APPROVED status.")
    logger.info(f"All artifacts saved in: {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Check for interactive flag
    interactive_mode = "--interactive" in sys.argv or "-i" in sys.argv
    
    # Get prompt from args or use default stress test
    prompt_args = [a for a in sys.argv[1:] if not a.startswith("-")]
    
    if prompt_args:
        test_prompt = " ".join(prompt_args)
    else:
        # ULTIMATE STRESS TEST: Centrifugal Compressor Impeller
        # test_prompt = (
        #     "Create a complex Centrifugal Compressor Impeller. "
        #     "1. The main hub is a truncated cone with a base diameter of 100mm (at Z=0), a top diameter of 30mm (at Z=60), and a total height of 60mm. "
        #     "2. The hub has a central bore hole of 15mm diameter going all the way through the Z axis for the driveshaft. "
        #     "3. On the surface of the hub, create 7 swept curved aerodynamic blades. "
        #     "4. Each blade should start at the base (radius 50mm) and curve upwards along the surface of the cone to the top (radius 15mm). "
        #     "5. The blades should have a uniform thickness of 2mm, an outward protrusion (height off the hub surface) of 15mm at the base, tapering to 5mm at the top. "
        #     "6. The blades should curve/twist around the Z axis by roughly 60 degrees from bottom to top to create the aerodynamic impeller shape. "
        #     "7. Ensure the final object is a single unified solid, assigned to a variable named 'result_solid'. "
        #     "DO NOT use hallucinated Selectors, stick to standard CadQuery operations like workplanes, extrude, sweep, or loft."
        # )

        test_prompt = (
            "Make a home AC unit, showing both pieces on different sides of the wall (inside and outside). The external piece should have a fan positioned on its external face vertically. Implement whatever features/methods you are missing in the script itself for your convenience. Use the simpler primitives when unsure."
        )

    
    run_pipeline(test_prompt, interactive=interactive_mode)