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
    """Sample wall thickness across the mesh surface using inward ray casting."""
    logger.info(f"Executing check_wall_thickness with threshold {min_thickness_mm}mm...")
    
    thin_regions = []
    min_found = float('inf')
    
    try:
        mesh_part = mrmesh.MeshPart(mesh)
        
        for face_id in range(min(mesh.topology.numValidFaces(), 500)):
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
                min_found = min(min_found, thickness)
                if thickness < min_thickness_mm:
                    thin_regions.append({
                        "face_id": face_id,
                        "thickness_mm": round(thickness, 3)
                    })
                    
        result = {
            "min_wall_thickness_mm": round(min_found, 3) if min_found != float('inf') else None,
            "thin_region_count": len(thin_regions),
            "passes_minimum": len(thin_regions) == 0,
            "thin_regions_sample": thin_regions[:5]
        }
        logger.debug(f"check_wall_thickness result: {result}")
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


def run_all_inspections(stl_filename: str, expected_dimensions: dict, output_dir: str) -> dict:
    """Wrapper function to execute all mesh checks and save a comprehensive report."""
    # Update logger to also write to output_dir
    global logger
    logger = get_agent_logger(f"{output_dir}/pipeline.log")
    
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
        
        # Determine overall validity
        overall_valid = (
            report["base_stats"]["is_watertight"] and
            not report["self_intersections"].get("has_self_intersections", True) and
            report["normals_consistency"].get("normals_consistent", False) and
            report["dimensions_check"].get("all_dimensions_pass", False)
        )
        report["overall_valid"] = overall_valid
        
        logger.info(f"Inspection complete. Overall Validity: {overall_valid}")
        
        # Save JSON report
        report_path = f"{output_dir}/inspection_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Saved inspection JSON report to {report_path}")
            
        return report

    except Exception as e:
        logger.critical(f"Critical error during comprehensive mesh inspection: {e}")
        return {"error": str(e)}
