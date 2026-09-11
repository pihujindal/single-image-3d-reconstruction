import open3d as o3d
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MESH_PATH = (
    BASE_DIR.parent
    / "phase4_mesh_cleaning"
    / "outputs"
    / "cleaned_mesh_final.ply"
)

POINT_CLOUD_PATH = (
    BASE_DIR.parent
    / "phase2_colored_point_cloud"
    / "outputs"
    / "filtered_point_cloud.npy"
)

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "refined_colored_mesh.ply"
)


# ============================================================
# LOAD CLEANED MESH
# ============================================================

mesh = o3d.io.read_triangle_mesh(
    str(MESH_PATH)
)

if mesh.is_empty():
    raise ValueError(
        "Phase 4 cleaned mesh is empty."
    )


print("=" * 70)
print("PHASE 5 - MESH REFINEMENT + COLOR TRANSFER")
print("=" * 70)

print("\nLoaded mesh:")
print(mesh)

print(
    "\nVertices:",
    len(mesh.vertices)
)

print(
    "Triangles:",
    len(mesh.triangles)
)


# ============================================================
# ORIGINAL MESH STATISTICS
# ============================================================

vertices = np.asarray(
    mesh.vertices
)

x = vertices[:, 0]
y = vertices[:, 1]
z = vertices[:, 2]

print("\n" + "=" * 70)
print("5.1 - ORIGINAL CLEANED MESH")
print("=" * 70)

print("X Min:", np.min(x))
print("X Max:", np.max(x))

print("Y Min:", np.min(y))
print("Y Max:", np.max(y))

print("Z Min:", np.min(z))
print("Z Max:", np.max(z))

print(
    "Surface Area:",
    mesh.get_surface_area()
)

print(
    "Edge Manifold:",
    mesh.is_edge_manifold()
)

print(
    "Vertex Manifold:",
    mesh.is_vertex_manifold()
)


# ============================================================
# MESH REFINEMENT
# ============================================================

print("\n" + "=" * 70)
print("5.2 - TAUBIN SMOOTHING")
print("=" * 70)

refined_mesh = mesh.__copy__()

refined_mesh = (
    refined_mesh.filter_smooth_taubin(
        number_of_iterations=5
    )
)


print(
    "Refined vertices:",
    len(refined_mesh.vertices)
)

print(
    "Refined triangles:",
    len(refined_mesh.triangles)
)


# ============================================================
# REFINED MESH STATISTICS
# ============================================================

refined_vertices = np.asarray(
    refined_mesh.vertices
)

refined_x = refined_vertices[:, 0]
refined_y = refined_vertices[:, 1]
refined_z = refined_vertices[:, 2]

print("\nRefined Bounding Box:")

print(
    "X Min:",
    np.min(refined_x)
)

print(
    "X Max:",
    np.max(refined_x)
)

print(
    "Y Min:",
    np.min(refined_y)
)

print(
    "Y Max:",
    np.max(refined_y)
)

print(
    "Z Min:",
    np.min(refined_z)
)

print(
    "Z Max:",
    np.max(refined_z)
)

print(
    "\nRefined Surface Area:",
    refined_mesh.get_surface_area()
)

print(
    "Refined Edge Manifold:",
    refined_mesh.is_edge_manifold()
)

print(
    "Refined Vertex Manifold:",
    refined_mesh.is_vertex_manifold()
)


# ============================================================
# COMPUTE NORMALS
# ============================================================

refined_mesh.compute_vertex_normals()

normals = np.asarray(
    refined_mesh.vertex_normals
)

print("\n" + "=" * 70)
print("5.3 - NORMAL VALIDATION")
print("=" * 70)

print(
    "Normals shape:",
    normals.shape
)

print(
    "NaN values:",
    np.isnan(normals).sum()
)

print(
    "Inf values:",
    np.isinf(normals).sum()
)

magnitude = np.linalg.norm(
    normals,
    axis=1
)

print(
    "Magnitude Min:",
    np.min(magnitude)
)

print(
    "Magnitude Max:",
    np.max(magnitude)
)

print(
    "Magnitude Mean:",
    np.mean(magnitude)
)


# ============================================================
# LOAD ORIGINAL COLORED POINT CLOUD
# ============================================================

print("\n" + "=" * 70)
print("5.4 - LOADING ORIGINAL RGB DATA")
print("=" * 70)

filtered_points = np.load(
    str(POINT_CLOUD_PATH)
)

print(
    "Point cloud shape:",
    filtered_points.shape
)

if filtered_points.shape[1] < 6:

    raise ValueError(
        "filtered_point_cloud.npy must contain "
        "XYZ + RGB (6 columns)."
    )


# ============================================================
# EXTRACT XYZ + RGB
# ============================================================

point_xyz = (
    filtered_points[:, :3]
    .astype(np.float64)
)

point_rgb = (
    filtered_points[:, 3:6]
    .astype(np.float64)
)


print(
    "Point XYZ shape:",
    point_xyz.shape
)

print(
    "Point RGB shape:",
    point_rgb.shape
)


# ============================================================
# REMOVE INVALID POINTS
# ============================================================

valid_points = np.all(
    np.isfinite(point_xyz),
    axis=1
)

valid_colors = np.all(
    np.isfinite(point_rgb),
    axis=1
)

valid_mask = (
    valid_points
    &
    valid_colors
)

point_xyz = point_xyz[
    valid_mask
]

point_rgb = point_rgb[
    valid_mask
]


print(
    "\nValid colored points:",
    len(point_xyz)
)


# ============================================================
# CREATE SOURCE POINT CLOUD
# ============================================================

source_pcd = o3d.geometry.PointCloud()

source_pcd.points = (
    o3d.utility.Vector3dVector(
        point_xyz
    )
)

source_pcd.colors = (
    o3d.utility.Vector3dVector(
        point_rgb / 255.0
    )
)


# ============================================================
# BUILD KD-TREE
# ============================================================

print("\nBuilding KD-tree...")

source_tree = o3d.geometry.KDTreeFlann(
    source_pcd
)


# ============================================================
# COLOR TRANSFER
# ============================================================

print("\n" + "=" * 70)
print("5.5 - RGB COLOR TRANSFER")
print("=" * 70)

mesh_vertices = np.asarray(
    refined_mesh.vertices
)

mesh_colors = np.zeros(
    (len(mesh_vertices), 3),
    dtype=np.float64
)


for i, vertex in enumerate(
    mesh_vertices
):

    [k, idx, distances] = (
        source_tree.search_knn_vector_3d(
            vertex,
            3
        )
    )

    if k > 0:

        distances = np.asarray(
            distances
        )

        idx = np.asarray(
            idx
        )

        # Inverse-distance weighting
        weights = (
            1.0
            /
            (distances + 1e-8)
        )

        weights = (
            weights
            /
            np.sum(weights)
        )

        neighbor_colors = (
            point_rgb[idx]
            / 255.0
        )

        mesh_colors[i] = (
            np.sum(
                neighbor_colors
                *
                weights[:, None],
                axis=0
            )
        )


# ============================================================
# ASSIGN COLORS TO MESH
# ============================================================

refined_mesh.vertex_colors = (
    o3d.utility.Vector3dVector(
        mesh_colors
    )
)


# ============================================================
# COLOR VALIDATION
# ============================================================

print(
    "Vertex colors:",
    len(refined_mesh.vertex_colors)
)

print(
    "Expected colors:",
    len(refined_mesh.vertices)
)

colors = np.asarray(
    refined_mesh.vertex_colors
)

print(
    "Color NaN:",
    np.isnan(colors).sum()
)

print(
    "Color Inf:",
    np.isinf(colors).sum()
)

print(
    "Color Min:",
    colors.min()
)

print(
    "Color Max:",
    colors.max()
)


# ============================================================
# SAVE COLORED MESH
# ============================================================

print("\n" + "=" * 70)
print("5.6 - SAVING COLORED MESH")
print("=" * 70)

success = o3d.io.write_triangle_mesh(
    str(OUTPUT_PATH),
    refined_mesh,
    write_vertex_normals=True,
    write_vertex_colors=True
)

print(
    "Saved:",
    success
)

print(
    "Output:",
    OUTPUT_PATH
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("5.7 - FINAL VALIDATION")
print("=" * 70)

print(
    "Vertices:",
    len(refined_mesh.vertices)
)

print(
    "Triangles:",
    len(refined_mesh.triangles)
)

print(
    "Colors:",
    len(refined_mesh.vertex_colors)
)

print(
    "Normals:",
    len(refined_mesh.vertex_normals)
)

print(
    "Surface Area:",
    refined_mesh.get_surface_area()
)

print(
    "Edge Manifold:",
    refined_mesh.is_edge_manifold()
)

print(
    "Vertex Manifold:",
    refined_mesh.is_vertex_manifold()
)


if (
    len(refined_mesh.vertices) > 0
    and
    len(refined_mesh.triangles) > 0
    and
    len(refined_mesh.vertex_colors)
    ==
    len(refined_mesh.vertices)
    and
    np.isnan(colors).sum() == 0
    and
    np.isinf(colors).sum() == 0
):

    print(
        "\nFINAL VALIDATION: PASS"
    )

else:

    print(
        "\nFINAL VALIDATION: CHECK REQUIRED"
    )


# ============================================================
# VISUALIZE
# ============================================================

print(
    "\nOpening colored refined mesh..."
)

o3d.visualization.draw_geometries(
    [refined_mesh],
    window_name="Refined Colored 3D Model"
)


print("\n" + "=" * 70)
print("PHASE 5 COMPLETE")
print("=" * 70)