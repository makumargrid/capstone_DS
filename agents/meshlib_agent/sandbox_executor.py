import sys
import os
import json
import subprocess
import tempfile

def run_invariant_baseline(mesh_path: str) -> dict:
    """
    Hardcoded safety checks that always run before the AI agent.
    These run in the MAIN process (safe reads only, no heavy OCCT writes).
    """
    try:
        import meshlib.mrmeshpy as mrmesh
    except ImportError as e:
        return {
            "load_failed": True,
            "passed": False,
            "hard_failures": [f"Failed to import meshlib: {str(e)}"],
            "is_watertight": False,
            "hole_count": 0,
            "volume": 0.0,
            "self_intersections": 0,
            "dimensions": {"x": 0.0, "y": 0.0, "z": 0.0},
            "num_faces": 0,
            "num_verts": 0
        }

    try:
        mesh = mrmesh.loadMesh(mesh_path)
    except Exception as e:
        return {
            "load_failed": True,
            "passed": False,
            "hard_failures": [f"Failed to load mesh: {str(e)}"],
            "is_watertight": False,
            "hole_count": 0,
            "volume": 0.0,
            "self_intersections": 0,
            "dimensions": {"x": 0.0, "y": 0.0, "z": 0.0},
            "num_faces": 0,
            "num_verts": 0
        }

    if not mesh or mesh.topology.numValidFaces() == 0:
        return {
            "load_failed": True,
            "passed": False,
            "hard_failures": ["Loaded mesh is empty or invalid"],
            "is_watertight": False,
            "hole_count": 0,
            "volume": 0.0,
            "self_intersections": 0,
            "dimensions": {"x": 0.0, "y": 0.0, "z": 0.0},
            "num_faces": 0,
            "num_verts": 0
        }

    try:
        # 1. Watertightness
        is_watertight = mesh.topology.isClosed()

        # 2. Holes count
        holes_edges = mesh.topology.findHoleRepresentiveEdges()
        try:
            hole_count = len(holes_edges)
        except TypeError:
            try:
                hole_count = holes_edges.size()
            except Exception:
                hole_count = 0

        # 3. Volume
        volume = mesh.volume()

        # 4. Self intersections count
        if hasattr(mrmesh, "localFindSelfIntersections"):
            intersecting_faces = mrmesh.localFindSelfIntersections(mesh)
            self_intersections = intersecting_faces.count()
        elif hasattr(mrmesh, "findSelfIntersections"):
            intersecting_faces = mrmesh.findSelfIntersections(mesh)
            self_intersections = intersecting_faces.count()
        else:
            self_intersections = 0

        # 5. Bounding box dimensions
        box = mesh.computeBoundingBox()
        x_dim = float(box.max.x - box.min.x)
        y_dim = float(box.max.y - box.min.y)
        z_dim = float(box.max.z - box.min.z)

        # 6. Topology counts
        num_faces = mesh.topology.numValidFaces()
        num_verts = mesh.topology.numValidVerts()

        # Hard failure conditions: not watertight, volume <= 0, self_intersections > 0
        hard_failures = []
        if not is_watertight:
            hard_failures.append("Mesh is not watertight")
        if volume <= 0:
            hard_failures.append(f"Mesh volume is <= 0 ({volume})")
        if self_intersections > 0:
            hard_failures.append(f"Mesh has self-intersections (intersecting faces: {self_intersections})")

        passed = len(hard_failures) == 0

        return {
            "load_failed": False,
            "is_watertight": is_watertight,
            "hole_count": hole_count,
            "volume": volume,
            "self_intersections": self_intersections,
            "dimensions": {
                "x": x_dim,
                "y": y_dim,
                "z": z_dim
            },
            "num_faces": num_faces,
            "num_verts": num_verts,
            "hard_failures": hard_failures,
            "passed": passed
        }
    except Exception as e:
        return {
            "load_failed": False,
            "passed": False,
            "hard_failures": [f"Error running baseline checks: {str(e)}"],
            "is_watertight": False,
            "hole_count": 0,
            "volume": 0.0,
            "self_intersections": 0,
            "dimensions": {"x": 0.0, "y": 0.0, "z": 0.0},
            "num_faces": 0,
            "num_verts": 0
        }

def run_in_sandbox(generated_code: str, mesh_path: str, timeout: int = 90) -> dict:
    """
    Execute AI-generated MeshLib code in complete isolation.
    """
    wrapper_content = f"""import json
import meshlib.mrmeshpy as mrmesh

mesh_path = {repr(mesh_path)}
mesh = mrmesh.loadMesh(mesh_path)
check_results = []

# --- BEGIN GENERATED CODE ---
{generated_code}
# --- END GENERATED CODE ---

print(json.dumps(check_results))
"""

    temp_file = None
    success = False
    check_results = []
    stderr_out = None
    exit_code = 0
    crash_type = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(wrapper_content)
            temp_file = f.name

        res = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        exit_code = res.returncode
        stdout = res.stdout
        stderr_out = res.stderr

        if stderr_out:
            stderr_out = stderr_out[-1500:]

        if exit_code == 0:
            try:
                check_results = json.loads(stdout.strip())
                if isinstance(check_results, list):
                    success = True
                else:
                    success = False
                    crash_type = "LOGIC_ERROR"
            except Exception:
                success = False
                crash_type = "LOGIC_ERROR"
        elif exit_code == -11:
            success = False
            crash_type = "SEGFAULT"
        else:
            success = False
            crash_type = "LOGIC_ERROR"

    except subprocess.TimeoutExpired as te:
        success = False
        crash_type = "TIMEOUT"
        exit_code = -15  # SIGTERM
        stderr_out = te.stderr if te.stderr else ""
        if isinstance(stderr_out, bytes):
            stderr_out = stderr_out.decode('utf-8', errors='replace')
        if stderr_out:
            stderr_out = stderr_out[-1500:]
    except Exception as e:
        success = False
        crash_type = "LOGIC_ERROR"
        stderr_out = f"Failed to execute wrapper: {str(e)}"
        exit_code = -1
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

    return {
        "success": success,
        "check_results": check_results,
        "stderr": stderr_out if stderr_out else None,
        "exit_code": exit_code,
        "crash_type": crash_type,
        "generated_code": generated_code
    }
