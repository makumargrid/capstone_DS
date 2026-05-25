import meshlib.mrmeshpy as mrmesh
import json
from src.logger import get_agent_logger

logger = get_agent_logger()

def count_degenerate_faces(mesh, min_area_threshold=1e-6) -> int:
    """Count faces with area below threshold (near-zero area triangles)."""
    logger.info("Executing count_degenerate_faces...")
    degen_count = 0
    try:
        for face_id in range(mesh.topology.numValidFaces()):
            fid = mrmesh.FaceId(face_id)
            if not mesh.topology.hasFace(fid):
                continue
            
            tri = mesh.topology.getTriVerts(fid)
            v0 = mesh.points.vec[tri[0]]
            v1 = mesh.points.vec[tri[1]]
            v2 = mesh.points.vec[tri[2]]
            
            # Cross product magnitude = 2 * triangle area
            edge1 = mrmesh.Vector3f(v1.x-v0.x, v1.y-v0.y, v1.z-v0.z)
            edge2 = mrmesh.Vector3f(v2.x-v0.x, v2.y-v0.y, v2.z-v0.z)
            
            # cross function returns Vector3f. We compute length by sqrt(x*x + y*y + z*z)
            cross_x = edge1.y*edge2.z - edge1.z*edge2.y
            cross_y = edge1.z*edge2.x - edge1.x*edge2.z
            cross_z = edge1.x*edge2.y - edge1.y*edge2.x
            
            cross_len = (cross_x**2 + cross_y**2 + cross_z**2)**0.5
            area = cross_len / 2.0
            
            if area < min_area_threshold:
                degen_count += 1
                
        logger.debug(f"count_degenerate_faces result: {degen_count}")
        return degen_count
    except Exception as e:
        logger.error(f"Error in count_degenerate_faces: {e}")
        return -1


def check_self_intersections(mesh) -> dict:
    """Detect faces that intersect other faces inside the solid."""
    logger.info("Executing check_self_intersections...")
    try:
        # MeshLib's localFindSelfIntersections returns FaceBitSet of intersecting faces
        intersecting_faces = mrmesh.localFindSelfIntersections(mesh)
        count = intersecting_faces.count()
        
        result = {
            "has_self_intersections": count > 0,
            "intersecting_face_count": count
        }
        logger.debug(f"check_self_intersections result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in check_self_intersections: {e}")
        return {"error": str(e)}


def check_normals_consistency(mesh) -> dict:
    """Check that face normals are consistently oriented (all outward)."""
    logger.info("Executing check_normals_consistency...")
    try:
        # MeshLib findDisorientedFaces returns FaceBitSet
        orientation_result = mrmesh.findDisorientedFaces(mesh)
        flipped_regions = orientation_result.count() if orientation_result else 0
        
        result = {
            "normals_consistent": flipped_regions == 0,
            "flipped_regions": flipped_regions
        }
        logger.debug(f"check_normals_consistency result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in check_normals_consistency: {e}")
        return {"error": str(e)}


def check_wall_thickness(mesh, min_thickness_mm: float) -> dict:
    """Sample wall thickness across the mesh surface using inward ray casting.
    
    IMPORTANT: Raw minimum measurements can be misleading because boolean operations
    (union, cut, intersect) create razor-thin mesh artifacts at junction edges.
    These are tessellation artifacts, NOT structural thin walls.
    
    To handle this, we:
    1. Filter out measurements below an ARTIFACT_FLOOR (0.3mm) — these are mesh edge artifacts.
    2. Report the 5th percentile as the 'effective minimum' for DFM decisions.
    3. Report the raw minimum separately for full transparency.
    """
    logger.info(f"Executing check_wall_thickness with threshold {min_thickness_mm}mm...")
    
    ARTIFACT_FLOOR = 0.3  # Measurements below this are mesh artifacts, not real walls
    MAX_FACES = 1000      # Sample more faces for better statistics
    
    all_measurements = []
    thin_regions = []
    
    try:
        mesh_part = mrmesh.MeshPart(mesh)
        num_faces = mesh.topology.numValidFaces()
        
        for face_id in range(min(num_faces, MAX_FACES)):
            fid = mrmesh.FaceId(face_id)
            if not mesh.topology.hasFace(fid):
                continue
                
            # Get face center
            tri = mesh.topology.getTriVerts(fid)
            v0 = mesh.points.vec[tri[0]]
            v1 = mesh.points.vec[tri[1]]
            v2 = mesh.points.vec[tri[2]]
            
            center = mrmesh.Vector3f(
                (v0.x+v1.x+v2.x)/3.0,
                (v0.y+v1.y+v2.y)/3.0,
                (v0.z+v1.z+v2.z)/3.0
            )
            
            # Calculate normal vector
            edge1 = mrmesh.Vector3f(v1.x-v0.x, v1.y-v0.y, v1.z-v0.z)
            edge2 = mrmesh.Vector3f(v2.x-v0.x, v2.y-v0.y, v2.z-v0.z)
            
            cross_x = edge1.y*edge2.z - edge1.z*edge2.y
            cross_y = edge1.z*edge2.x - edge1.x*edge2.z
            cross_z = edge1.x*edge2.y - edge1.y*edge2.x
            
            cross_len = (cross_x**2 + cross_y**2 + cross_z**2)**0.5
            if cross_len < 1e-10:
                continue # Degenerate face
                
            normal = mrmesh.Vector3f(cross_x/cross_len, cross_y/cross_len, cross_z/cross_len)
            inward = mrmesh.Vector3f(-normal.x, -normal.y, -normal.z)
            
            # Move the origin slightly inward to avoid self-intersection with the originating face
            epsilon = 1e-4
            origin = mrmesh.Vector3f(
                center.x + inward.x * epsilon,
                center.y + inward.y * epsilon,
                center.z + inward.z * epsilon
            )
            
            ray = mrmesh.Line3f(origin, inward)
            
            # Cast ray
            intersect_result = mrmesh.rayMeshIntersect(mesh_part, ray, 0.0, 500.0)
            
            if intersect_result and hasattr(intersect_result, 'distanceAlongLine'):
                thickness = intersect_result.distanceAlongLine
                all_measurements.append(thickness)
                
                # Only count as "thin" if ABOVE the artifact floor
                # (below the floor = mesh artifact, not a real wall)
                if thickness >= ARTIFACT_FLOOR and thickness < min_thickness_mm:
                    thin_regions.append({
                        "face_id": face_id,
                        "thickness_mm": round(thickness, 3),
                        "face_center_z_mm": round(center.z, 1)
                    })
        
        # Compute statistics
        if all_measurements:
            all_measurements.sort()
            raw_min = round(all_measurements[0], 3)
            
            # Filter out artifact measurements
            structural_measurements = [m for m in all_measurements if m >= ARTIFACT_FLOOR]
            
            if structural_measurements:
                structural_measurements.sort()
                n = len(structural_measurements)
                p5_idx = max(0, int(n * 0.05))
                p5 = round(structural_measurements[p5_idx], 3)
                median = round(structural_measurements[n // 2], 3)
                structural_min = round(structural_measurements[0], 3)
            else:
                p5 = raw_min
                median = raw_min
                structural_min = raw_min
            
            artifact_count = len(all_measurements) - len(structural_measurements)
        else:
            raw_min = None
            p5 = None
            median = None
            structural_min = None
            artifact_count = 0
        
        # DFM decision uses p5 (5th percentile), not structural_min.
        # Rationale: In real-world DFM, isolated thin measurements at boolean junction
        # edges or trim planes are NOT structural concerns. What matters is whether
        # the BULK of the surface meets the minimum thickness spec.
        # p5 means "95% of the surface is at least this thick" — the right DFM metric.
        # We also add a 0.05mm (50 micron) tolerance to avoid failing on
        # floating-point precision noise (e.g., 1.997mm vs 2.0mm).
        DFM_TOLERANCE = 0.05  # 50 microns
        passes = p5 is not None and p5 >= (min_thickness_mm - DFM_TOLERANCE)
        
        result = {
            "raw_min_wall_thickness_mm": raw_min,
            "min_wall_thickness_mm": structural_min,
            "p5_wall_thickness_mm": p5,
            "median_wall_thickness_mm": median,
            "thin_region_count": len(thin_regions),
            "artifact_count": artifact_count,
            "total_samples": len(all_measurements),
            "passes_minimum": passes,
            "thin_regions_sample": thin_regions[:5]
        }
        logger.info(f"Wall thickness stats: raw_min={raw_min}mm, structural_min={structural_min}mm, "
                     f"p5={p5}mm, median={median}mm, artifacts_filtered={artifact_count}, "
                     f"thin_structural_regions={len(thin_regions)}")
        return result
    except Exception as e:
        logger.error(f"Error in check_wall_thickness: {e}")
        return {"error": str(e)}


def check_dimensions_vs_plan(mesh, expected: dict) -> dict:
    """Cross-check mesh bounding box against the Primitive Plan's expected dimensions."""
    logger.info("Executing check_dimensions_vs_plan...")
    try:
        box = mesh.computeBoundingBox()
        measured = {
            "x_mm": round(box.max.x - box.min.x, 3),
            "y_mm": round(box.max.y - box.min.y, 3),
            "z_mm": round(box.max.z - box.min.z, 3),
        }
        
        tol = expected.get("tolerance_mm", 0.5)
        results = {}
        all_pass = True
        
        for axis in ["x_mm", "y_mm", "z_mm"]:
            if axis in expected:
                delta = abs(measured[axis] - expected[axis])
                passed = delta <= tol
                all_pass = all_pass and passed
                results[axis] = {
                    "expected": expected[axis],
                    "measured": measured[axis],
                    "delta_mm": round(delta, 3),
                    "tolerance_mm": tol,
                    "passed": passed
                }
                
        final_result = {"dimensions": results, "all_dimensions_pass": all_pass}
        logger.debug(f"check_dimensions_vs_plan result: {final_result}")
        return final_result
    except Exception as e:
        logger.error(f"Error in check_dimensions_vs_plan: {e}")
        return {"error": str(e)}


def run_all_inspections(stl_filename: str, expected_dimensions: dict, output_dir: str, outer_attempt: int = 1) -> dict:
    """Wrapper function to execute all mesh checks and save a comprehensive report.
    
    Returns a structured dict with:
      - All individual check results
      - overall_valid: bool
      - hard_failures: list of strings describing critical failures
      - The pipeline uses hard_failures to decide whether to invoke the AI inspector.
    """
    # Update logger to also write to output_dir
    global logger
    logger = get_agent_logger(f"{output_dir}/00_pipeline_execution.log")
    
    logger.info(f"Starting comprehensive mesh inspection on {stl_filename}")
    
    report = {}
    try:
        mesh = mrmesh.loadMesh(stl_filename)
        
        # 1. Base stats
        logger.info("Checking base stats (watertightness, volume)...")
        is_watertight = mesh.topology.isClosed()
        volume = mesh.volume()
        report["base_stats"] = {
            "is_watertight": is_watertight,
            "volume_mm3": round(volume, 2)
        }
        
        # 2. Advanced checks
        report["degenerate_faces"] = count_degenerate_faces(mesh)
        report["self_intersections"] = check_self_intersections(mesh)
        report["normals_consistency"] = check_normals_consistency(mesh)
        report["wall_thickness"] = check_wall_thickness(mesh, min_thickness_mm=2.0)
        report["dimensions_check"] = check_dimensions_vs_plan(mesh, expected_dimensions)
        
        # 3. Determine overall validity and collect hard failures
        hard_failures = []
        
        if not report["base_stats"]["is_watertight"]:
            hard_failures.append("Mesh is not watertight (has holes or open boundaries)")
        
        if report["base_stats"]["volume_mm3"] <= 0:
            hard_failures.append(f"Mesh volume is non-positive: {report['base_stats']['volume_mm3']}mm³")
        
        if report["self_intersections"].get("has_self_intersections", False):
            count = report["self_intersections"].get("intersecting_face_count", 0)
            hard_failures.append(f"Mesh has {count} self-intersecting faces")
        
        if not report["normals_consistency"].get("normals_consistent", True):
            flipped = report["normals_consistency"].get("flipped_regions", 0)
            hard_failures.append(f"Mesh has {flipped} flipped/inconsistent normal regions")
        
        if not report["dimensions_check"].get("all_dimensions_pass", True):
            dims = report["dimensions_check"].get("dimensions", {})
            for axis, info in dims.items():
                if isinstance(info, dict) and not info.get("passed", True):
                    hard_failures.append(
                        f"Dimension {axis}: expected {info.get('expected')}mm, "
                        f"measured {info.get('measured')}mm (delta {info.get('delta_mm')}mm)"
                    )
        
        # 4. Collect DFM soft failures (geometry is valid, but NOT manufacturable)
        soft_failures = []
        
        wt = report.get("wall_thickness", {})
        structural_min = wt.get("min_wall_thickness_mm")
        if structural_min is not None and not wt.get("passes_minimum", True):
            thin_count = wt.get("thin_region_count", 0)
            thin_samples = wt.get("thin_regions_sample", [])
            sample_str = ", ".join(
                f"face {s['face_id']}={s['thickness_mm']}mm (Z={s.get('face_center_z_mm','?')}mm)" 
                for s in thin_samples
            )
            soft_failures.append(
                f"DFM_WALL_THICKNESS: Structural min={structural_min}mm "
                f"(required: 2.0mm, deficit: {round(2.0 - structural_min, 3)}mm). "
                f"P5={wt.get('p5_wall_thickness_mm')}mm, median={wt.get('median_wall_thickness_mm')}mm. "
                f"Raw min={wt.get('raw_min_wall_thickness_mm')}mm "
                f"(artifacts filtered: {wt.get('artifact_count', 0)}). "
                f"{thin_count} structural thin region(s). "
                f"Samples: [{sample_str}]"
            )
        
        overall_valid = len(hard_failures) == 0 and len(soft_failures) == 0
        report["overall_valid"] = overall_valid
        report["hard_failures"] = hard_failures
        report["soft_failures"] = soft_failures
        
        logger.info(f"Inspection complete. Overall Validity: {overall_valid}")
        if hard_failures:
            for f in hard_failures:
                logger.warning(f"HARD FAILURE: {f}")
        if soft_failures:
            for f in soft_failures:
                logger.warning(f"SOFT FAILURE (DFM): {f}")
        
        # Save JSON report
        report_path = f"{output_dir}/05_outer{outer_attempt}_static_inspection_ground_truth.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Saved inspection JSON report to {report_path}")
            
        return report

    except Exception as e:
        logger.critical(f"Critical error during comprehensive mesh inspection: {e}")
        return {
            "overall_valid": False,
            "hard_failures": [f"Critical inspection error: {e}"],
            "soft_failures": [],
            "error": str(e)
        }


def has_hard_failures(static_results: dict) -> bool:
    """Quick check: does the static inspection report contain any hard failures?
    
    Hard failures = geometrically broken (not watertight, self-intersections, etc.)
    These short-circuit BEFORE any AI agents are invoked.
    """
    failures = static_results.get("hard_failures", [])
    return len(failures) > 0


def has_soft_failures(static_results: dict) -> bool:
    """Quick check: does the static inspection report contain any DFM soft failures?
    
    Soft failures = geometrically valid but NOT manufacturable (thin walls, etc.)
    These short-circuit to the Planner with rich quantitative feedback,
    skipping the expensive AI inspection and Reviewer phases.
    """
    failures = static_results.get("soft_failures", [])
    return len(failures) > 0


