import sys
import os
# Force google-genai / google-adk to use Developer API instead of Vertex AI
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'false'

import json
import time
import datetime
import uuid
import glob
import asyncio
from dotenv import load_dotenv

# Ensure workspace root is in sys.path for absolute imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Load environment variables
load_dotenv(os.path.join(parent_dir, ".env"))

from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

from agents.meshlib_agent.agent import root_agent
from agents.meshlib_agent.sandbox_executor import run_invariant_baseline

# =====================================================================
# Section 1: Event classifier
# =====================================================================
def classify_event(event) -> str:
    """Classifies an ADK runner event into one of the known types."""
    if not event or not getattr(event, "content", None):
        return "OTHER"
    
    parts = getattr(event.content, "parts", None)
    if not parts:
        return "OTHER"
        
    for part in parts:
        try:
            if getattr(part, "function_call", None) is not None:
                return "TOOL_CALL"
            if getattr(part, "function_response", None) is not None:
                return "TOOL_RESULT"
            if getattr(part, "text", None) is not None:
                author = getattr(event, "author", "").lower()
                if "user" in author or "pipeline" in author:
                    return "USER_MESSAGE"
                else:
                    return "AGENT_TEXT"
        except Exception:
            pass
            
    return "OTHER"

# =====================================================================
# Section 2: Event printer
# =====================================================================
def get_colors():
    """Returns ANSI escape characters for color terminal output if connected to a TTY."""
    if sys.stdout.isatty():
        return {
            "yellow": "\033[93m",
            "green": "\033[92m",
            "red": "\033[91m",
            "cyan": "\033[96m",
            "white_bold": "\033[1m\033[97m",
            "reset": "\033[0m",
            "bold": "\033[1m"
        }
    else:
        return {
            "yellow": "",
            "green": "",
            "red": "",
            "cyan": "",
            "white_bold": "",
            "reset": "",
            "bold": ""
        }

def print_event(event, seq: int, elapsed: float):
    """Prints a single event with formatted style and color coding."""
    C = get_colors()
    event_type = classify_event(event)
    
    if event_type == "OTHER":
        return
        
    parts = event.content.parts
    for part in parts:
        try:
            if event_type == "TOOL_CALL":
                fc = part.function_call
                args = fc.args or {}
                desc = args.get("description") or args.get("reason") or "No description provided."
                script = args.get("script_content") or ""
                lines = script.splitlines()
                
                print("\n" + "=" * 80)
                print(f"{C['yellow']}[STEP {seq}] TOOL CALL  ── {fc.name}  (elapsed: {elapsed:.1f}s){C['reset']}")
                print(f"Description: {desc}")
                print("-" * 40)
                for idx, line in enumerate(lines, 1):
                    print(f"{idx:4d} | {line}")
                print("-" * 40)
                print(f"Total lines: {len(lines)}")
                print("=" * 80 + "\n")
                
            elif event_type == "TOOL_RESULT":
                fr = part.function_response
                response = fr.response or {}
                
                # Unwrap output if nested
                if "result" in response and isinstance(response["result"], dict):
                    res = response["result"]
                elif "output" in response and isinstance(response["output"], dict):
                    res = response["output"]
                else:
                    res = response
                    
                success = res.get("success", False)
                crash_type = res.get("crash_type")
                exit_code = res.get("exit_code")
                check_results = res.get("check_results") or []
                stderr = res.get("stderr") or ""
                
                color = C["green"] if success else C["red"]
                print("\n" + "=" * 80)
                print(f"{color}[STEP {seq}] TOOL RESULT  (elapsed: {elapsed:.1f}s){C['reset']}")
                print(f"Success: {success}")
                if crash_type:
                    print(f"Crash Type: {crash_type}")
                if exit_code is not None:
                    print(f"Exit Code: {exit_code}")
                
                if success:
                    print("\nCheck Results:")
                    for cr in check_results:
                        passed = cr.get("passed", False)
                        symbol = "✓" if passed else "✗"
                        status_color = C["green"] if passed else C["red"]
                        line = f"  {status_color}{symbol} {cr.get('check_name')}: Measured={cr.get('measured')} {cr.get('unit', '')}, Expected={cr.get('expected')}"
                        if not passed and cr.get("reason"):
                            line += f" (Reason: {cr.get('reason')})"
                        line += C["reset"]
                        print(line)
                else:
                    print(f"\n{C['red']}{C['bold']}CRASH DETECTED: {crash_type}{C['reset']}")
                    if stderr:
                        print("-" * 40)
                        print(stderr[:500] + ("..." if len(stderr) > 500 else ""))
                        print("-" * 40)
                print("=" * 80 + "\n")
                
            elif event_type == "AGENT_TEXT":
                text = part.text or ""
                is_final = False
                try:
                    is_final = event.is_final_response()
                except Exception:
                    pass
                    
                if is_final:
                    print(f"\n{C['white_bold']}[STEP {seq}] FINAL VERDICT  (elapsed: {elapsed:.1f}s){C['reset']}")
                    # Attempt to pretty print JSON
                    cleaned = text.replace("```json", "").replace("```", "").strip()
                    try:
                        verdict_json = json.loads(cleaned)
                        print(f"{C['green']}{json.dumps(verdict_json, indent=4)}{C['reset']}")
                    except Exception:
                        print(text)
                    print()
                else:
                    print(f"\n{C['cyan']}[STEP {seq}] AGENT REASONING  (elapsed: {elapsed:.1f}s){C['reset']}")
                    truncated = text[:400]
                    if len(text) > 400:
                        truncated += "..."
                    print(truncated)
                    print()
                    
            elif event_type == "USER_MESSAGE":
                text = part.text or ""
                print(f"\n[STEP {seq}] USER INPUT  (elapsed: {elapsed:.1f}s)")
                truncated = text[:200]
                if len(text) > 200:
                    truncated += "..."
                print(truncated)
                print()
                
        except Exception as e:
            print(f"{C['red']}[STEP {seq}] UNKNOWN PART: {e}{C['reset']}")

# =====================================================================
# Section 4: Artifact helpers
# =====================================================================
def event_to_dict(event) -> dict:
    """Safely dumps a Pydantic event structure to JSON-compatible dict."""
    if hasattr(event, "model_dump"):
        try:
            return event.model_dump(mode="json")
        except Exception:
            pass
    if hasattr(event, "dict"):
        try:
            return event.dict()
        except Exception:
            pass
    try:
        return dict(event)
    except Exception:
        return {"raw_event_str": str(event)}

def get_event_summary(event_type: str, event) -> str:
    """Creates a brief textual summary of the event content."""
    try:
        parts = event.content.parts
        if not parts:
            return "N/A"
        part = parts[0]
        if event_type == "TOOL_CALL":
            return f"Function Call: {part.function_call.name}"
        elif event_type == "TOOL_RESULT":
            response = part.function_response.response or {}
            success = response.get("success", response.get("output", {}).get("success", False))
            return f"Success: {success}"
        elif event_type in ("AGENT_TEXT", "USER_MESSAGE"):
            text = part.text or ""
            return text[:200] + ("..." if len(text) > 200 else "")
    except Exception:
        pass
    return "N/A"

def generate_markdown_transcript(seq_events) -> str:
    """Compiles a complete human-readable markdown transcript."""
    md = "# Agent Observation Transcript\n\n"
    for seq, event_type, elapsed, event in seq_events:
        md += f"## Step {seq} — [{event_type}]\n"
        md += f"**Time:** {elapsed:.2f}s\n"
        md += f"**Author:** {getattr(event, 'author', 'unknown')}\n\n"
        
        parts = getattr(event.content, "parts", None) or []
        for part in parts:
            try:
                if event_type == "TOOL_CALL":
                    fc = part.function_call
                    args = fc.args or {}
                    desc = args.get("description") or args.get("reason") or "N/A"
                    md += f"**Description:** {desc}\n\n"
                    md += "```python\n" + args.get("script_content", "") + "\n```\n\n"
                elif event_type == "TOOL_RESULT":
                    fr = part.function_response
                    response = fr.response or {}
                    if "result" in response and isinstance(response["result"], dict):
                        res = response["result"]
                    elif "output" in response and isinstance(response["output"], dict):
                        res = response["output"]
                    else:
                        res = response
                        
                    md += f"**Success:** {res.get('success', False)}\n"
                    if res.get("crash_type"):
                        md += f"**Crash Type:** {res.get('crash_type')}\n"
                    md += f"**Exit Code:** {res.get('exit_code')}\n\n"
                    
                    if res.get("success"):
                        md += "### Check Results:\n"
                        for cr in res.get("check_results", []):
                            symbol = "✓" if cr.get("passed") else "✗"
                            md += f"- **{symbol} {cr.get('check_name')}**: Measured={cr.get('measured')} {cr.get('unit', '')}, Expected={cr.get('expected')}"
                            if not cr.get("passed") and cr.get("reason"):
                                md += f" (Reason: {cr.get('reason')})"
                            md += "\n"
                        md += "\n"
                    else:
                        md += "### Stderr:\n```\n" + (res.get("stderr") or "") + "\n```\n\n"
                elif event_type == "AGENT_TEXT":
                    md += part.text + "\n\n"
                elif event_type == "USER_MESSAGE":
                    md += f"> {part.text}\n\n"
            except Exception as e:
                md += f"*Error formatting part: {e}*\n\n"
        md += "---\n\n"
    return md

# =====================================================================
# Main Orchestration Loop
# =====================================================================
def main():
    C = get_colors()
    print("=" * 80)
    print(f"{C['white_bold']}MeshLib Inspector Agent - Observation Console{C['reset']}")
    print("=" * 80)

    # 1. Parse Arguments & Test Files
    mesh_path = None
    if len(sys.argv) > 1:
        mesh_path = os.path.abspath(sys.argv[1])
    else:
        # Search recursively for the most recent STL under outputs/
        stl_files = glob.glob(os.path.join(parent_dir, "outputs/**/*.stl"), recursive=True)
        if stl_files:
            stl_files.sort(key=os.path.getmtime, reverse=True)
            mesh_path = os.path.abspath(stl_files[0])

    if not mesh_path or not os.path.exists(mesh_path):
        print(f"{C['red']}Error: No STL mesh file found or specified.{C['reset']}")
        sys.exit(1)

    # Load primitive plan
    primitive_plan = {
        "expected_dims": {
            "x_mm": 100.0,
            "y_mm": 80.0,
            "z_mm": 4.0,
            "tolerance_mm": 0.5
        },
        "min_wall_mm": 2.0,
        "manufacturing_process": "FDM_3D_print",
        "primitives": []
    }
    
    if len(sys.argv) > 2:
        try:
            primitive_plan = json.loads(sys.argv[2])
        except Exception as e:
            print(f"{C['red']}Error parsing primitive plan JSON argument: {e}{C['reset']}")
            sys.exit(1)

    print(f"Testing Mesh Path: {mesh_path}")
    print(f"Primitive Plan:    {json.dumps(primitive_plan)}")
    print("-" * 80)

    # 2. Timing Tracker Initial Setup
    times = {
        "baseline_check": 0.0,
        "agent_reasoning": 0.0,
        "tool_execution": 0.0,
        "total": 0.0
    }
    start_time = time.time()

    # 3. Run Invariant Baseline
    print("Executing baseline watertightness & geometry checks...")
    baseline_start = time.time()
    baseline = run_invariant_baseline(mesh_path)
    times["baseline_check"] = time.time() - baseline_start
    print(f"Baseline complete ({times['baseline_check']:.2f}s). watertight={baseline.get('is_closed')}")

    if baseline.get("load_failed"):
        print(f"{C['red']}Baseline check failed mesh load. Terminating.{C['reset']}")
        sys.exit(1)
    print("-" * 80)

    # 4. Prepare ADK Session
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

    runner = Runner(
        agent=root_agent,
        app_name="meshlib_agent",
        session_service=session_service
    )

    message = (
        f"Mesh Path: {mesh_path}\n\n"
        f"Primitive Plan:\n{json.dumps(primitive_plan, indent=2)}\n\n"
        f"Baseline Results:\n{json.dumps(baseline, indent=2)}\n\n"
        f"Instruction: The baseline has already checked watertightness, volume, "
        f"self-intersections, and bounding box — do not repeat these, "
        f"focus on plan-specific dimensional and feature verification."
    )
    content = types.Content(role='user', parts=[types.Part(text=message)])

    # 5. Process Events Loop (Real-time tracking)
    print(f"{C['bold']}Streaming ADK agent execution in real-time...{C['reset']}")
    
    seq_events = [] # stores (seq_num, event_type, elapsed_sec, event)
    last_event_time = time.time()
    tool_call_start = None
    
    # State tracking for summary
    tool_calls_count = 0
    tool_successes = 0
    tool_failures = 0
    repair_iterations = 0
    last_tool_success = None
    
    # Run the generator
    events_generator = runner.run(user_id="user", session_id=session_id, new_message=content)
    
    for event in events_generator:
        current_time = time.time()
        elapsed_sec = current_time - start_time
        
        event_type = classify_event(event)
        
        # Track reasoning times
        if event_type == "AGENT_TEXT":
            times["agent_reasoning"] += (current_time - last_event_time)
        elif event_type == "TOOL_CALL":
            tool_call_start = current_time
            times["agent_reasoning"] += (current_time - last_event_time)
            
            # Count tool calls and repair iterations
            tool_calls_count += 1
            if last_tool_success == False:
                repair_iterations += 1
                
        elif event_type == "TOOL_RESULT":
            if tool_call_start is not None:
                times["tool_execution"] += (current_time - tool_call_start)
                tool_call_start = None
            
            # Count successes vs failures
            parts = getattr(event.content, "parts", [])
            if parts:
                response = parts[0].function_response.response or {}
                if "result" in response and isinstance(response["result"], dict):
                    res = response["result"]
                elif "output" in response and isinstance(response["output"], dict):
                    res = response["output"]
                else:
                    res = response
                success = res.get("success", False)
                last_tool_success = success
                if success:
                    tool_successes += 1
                else:
                    tool_failures += 1
                    
        # Append to record list
        seq_num = len(seq_events) + 1
        seq_events.append((seq_num, event_type, elapsed_sec, event))
        
        # Print event in real-time
        print_event(event, seq_num, elapsed_sec)
        
        last_event_time = current_time

    # Calculate overall total time
    times["total"] = time.time() - start_time

    # 6. Parse Final Verdict
    overall_passed = False
    failure_class = None
    verdict = None
    
    # Locate final response
    for _, event_type, _, event in reversed(seq_events):
        if event_type == "AGENT_TEXT":
            try:
                if event.is_final_response():
                    text = event.content.parts[0].text or ""
                    cleaned = text.replace("```json", "").replace("```", "").strip()
                    verdict = json.loads(cleaned)
                    overall_passed = verdict.get("overall_passed", False)
                    failure_class = verdict.get("failure_class")
                    break
            except Exception:
                pass

    # 7. Print Timing Table
    print("=" * 80)
    print(f"{C['white_bold']}Timing Summary Table:{C['reset']}")
    print("-" * 40)
    print(f"| {'Phase':<20} | {'Duration (s)':<13} |")
    print("-" * 40)
    for phase, val in times.items():
        print(f"| {phase:<20} | {val:<11.2f}s |")
    print("-" * 40)
    print()

    # 8. Print Summary
    print("=" * 80)
    print(f"{C['white_bold']}Inspection Run Summary:{C['reset']}")
    print("-" * 40)
    print(f"Total Steps (Events):    {len(seq_events)}")
    print(f"Tool Calls Made:         {tool_calls_count}")
    print(f"Tool Successes:          {tool_successes}")
    print(f"Tool Failures:           {tool_failures}")
    print(f"Repair Iterations:       {repair_iterations}")
    print(f"Failure Classification:  {failure_class}")
    passed_str = f"{C['green']}TRUE{C['reset']}" if overall_passed else f"{C['red']}FALSE{C['reset']}"
    print(f"Overall Passed:          {passed_str}")
    print(f"Total Time Taken:        {times['total']:.2f}s")
    print("=" * 80)

    # 9. Save Artifacts
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(parent_dir, f"outputs/observation_run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    # Save baseline report
    with open(os.path.join(out_dir, "baseline_report.json"), "w") as f:
        json.dump(baseline, f, indent=4)
        
    # Save timing report
    with open(os.path.join(out_dir, "timing_report.json"), "w") as f:
        json.dump(times, f, indent=4)
        
    # Save agent trace
    trace = []
    for seq, etype, elapsed, event in seq_events:
        trace.append({
            "sequence_number": seq,
            "event_type": etype,
            "author": getattr(event, "author", "unknown"),
            "timestamp_ms": int(getattr(event, "timestamp", 0)),
            "elapsed_seconds": round(elapsed, 2),
            "content_summary": get_event_summary(etype, event),
            "full_content": event_to_dict(event)
        })
    with open(os.path.join(out_dir, "agent_trace.json"), "w") as f:
        json.dump(trace, f, indent=4)
        
    # Save markdown transcript
    transcript_md = generate_markdown_transcript(seq_events)
    with open(os.path.join(out_dir, "agent_transcript.md"), "w") as f:
        f.write(transcript_md)

    print(f"\nArtifacts successfully saved to: {out_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
