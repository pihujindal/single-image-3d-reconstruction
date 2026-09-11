import open3d as o3d
import numpy as np
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent

MESH_PATH = BASE_DIR.parent / "phase3" / "reconstructed_mesh.ply"
DENSITY_PATH = BASE_DIR.parent / "phase3" / "densities.npy"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "cleaned_mesh_final.ply"


mesh = o3d.io.read_triangle_mesh(str(MESH_PATH))

if mesh.is_empty():
    raise ValueError("Reconstructed mesh is empty.")

print("=" * 70)
print("PHASE 4 - MESH CLEANING")
print("=" * 70)

print("\nLoaded mesh:")
print(mesh)


vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

print("\n" + "=" * 70)
print("4.1 - BEFORE CLEANING")
print("=" * 70)

print("Vertices :", len(vertices))
print("Triangles:", len(triangles))

print("\nVertex dtype:", vertices.dtype)
print("Vertex shape:", vertices.shape)

print("\nNaN values:", np.isnan(vertices).sum())
print("Infinite values:", np.isinf(vertices).sum())


if len(vertices) > 0:

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    print("\nX Min:", x.min())
    print("X Max:", x.max())

    print("Y Min:", y.min())
    print("Y Max:", y.max())

    print("Z Min:", z.min())
    print("Z Max:", z.max())


print("\nSurface Area:", mesh.get_surface_area())

print("Edge Manifold:", mesh.is_edge_manifold())
print("Vertex Manifold:", mesh.is_vertex_manifold())


saved_densities = np.load(str(DENSITY_PATH))

print("\n" + "=" * 70)
print("4.2 - DENSITY ANALYSIS")
print("=" * 70)

print("Densities length:", len(saved_densities))
print("Densities shape :", saved_densities.shape)
print("Densities dtype :", saved_densities.dtype)

print("Density Min    :", np.min(saved_densities))
print("Density Max    :", np.max(saved_densities))
print("Density Mean   :", np.mean(saved_densities))
print("Density Median :", np.median(saved_densities))


if len(saved_densities) != len(vertices):

    raise ValueError(
        "\nDensity count does not match mesh vertex count.\n"
        f"Vertices  : {len(vertices)}\n"
        f"Densities : {len(saved_densities)}"
    )

print("\nDensity count matches mesh vertices.")



thresholds = [3, 4, 5, 6]

print("\nDensity threshold statistics:")

for threshold in thresholds:

    count = np.sum(
        saved_densities < threshold
    )

    percentage = (
        count / len(saved_densities)
    ) * 100

    print(
        f"Threshold {threshold}: "
        f"{count} vertices "
        f"({percentage:.2f}%)"
    )



DENSITY_THRESHOLD = 5.0

low_density_mask = (
    saved_densities < DENSITY_THRESHOLD
)

low_density_count = np.sum(
    low_density_mask
)

print(
    "\nSelected density threshold:",
    DENSITY_THRESHOLD
)

print(
    "Low-density vertices:",
    low_density_count
)

print(
    "Low-density percentage:",
    f"{low_density_count / len(vertices) * 100:.2f}%"
)


triangle_array = np.asarray(
    mesh.triangles
)

affected_triangles = np.any(
    low_density_mask[triangle_array],
    axis=1
)

affected_count = np.sum(
    affected_triangles
)

print("\nAffected triangles:", affected_count)

if len(triangle_array) > 0:

    affected_percentage = (
        affected_count /
        len(triangle_array)
    ) * 100

    print(
        "Affected triangle percentage:",
        f"{affected_percentage:.2f}%"
    )


cleaned_mesh = o3d.geometry.TriangleMesh(
    mesh
)


cleaned_mesh.remove_vertices_by_mask(
    low_density_mask
)

print("\n" + "=" * 70)
print("4.3 - AFTER DENSITY CLEANING")
print("=" * 70)

print(
    "Vertices :",
    len(cleaned_mesh.vertices)
)

print(
    "Triangles:",
    len(cleaned_mesh.triangles)
)

cleaned_mesh.remove_unreferenced_vertices()

print("\nAfter removing unreferenced vertices:")

print(
    "Vertices :",
    len(cleaned_mesh.vertices)
)

print(
    "Triangles:",
    len(cleaned_mesh.triangles)
)


cleaned_mesh.remove_degenerate_triangles()

print("\nAfter removing degenerate triangles:")

print(
    "Vertices :",
    len(cleaned_mesh.vertices)
)

print(
    "Triangles:",
    len(cleaned_mesh.triangles)
)

cleaned_mesh.remove_duplicated_triangles()

print("\nAfter removing duplicated triangles:")

print(
    "Vertices :",
    len(cleaned_mesh.vertices)
)

print(
    "Triangles:",
    len(cleaned_mesh.triangles)
)

cleaned_mesh.remove_duplicated_vertices()

print("\nAfter removing duplicated vertices:")

print(
    "Vertices :",
    len(cleaned_mesh.vertices)
)

print(
    "Triangles:",
    len(cleaned_mesh.triangles)
)

cleaned_mesh.remove_unreferenced_vertices()


cleaned_mesh.compute_vertex_normals()
cleaned_mesh.compute_triangle_normals()


normals = np.asarray(
    cleaned_mesh.vertex_normals
)

print("\n" + "=" * 70)
print("4.4 - NORMAL VALIDATION")
print("=" * 70)

print("Normals shape:", normals.shape)
print("Normals dtype:", normals.dtype)

print(
    "Normals NaN:",
    np.isnan(normals).sum()
)

print(
    "Normals Infinite:",
    np.isinf(normals).sum()
)


final_vertices = len(
    cleaned_mesh.vertices
)

final_triangles = len(
    cleaned_mesh.triangles
)


print("\n" + "=" * 70)
print("4.5 - FINAL MESH VALIDATION")
print("=" * 70)

print("\nFinal Vertices :", final_vertices)
print("Final Triangles:", final_triangles)


if final_vertices > 0:

    bbox = (
        cleaned_mesh
        .get_axis_aligned_bounding_box()
    )

    x_min = bbox.min_bound[0]
    x_max = bbox.max_bound[0]

    y_min = bbox.min_bound[1]
    y_max = bbox.max_bound[1]

    z_min = bbox.min_bound[2]
    z_max = bbox.max_bound[2]

    print("\nBounding Box:")

    print("X Min:", x_min)
    print("X Max:", x_max)

    print("Y Min:", y_min)
    print("Y Max:", y_max)

    print("Z Min:", z_min)
    print("Z Max:", z_max)


surface_area = (
    cleaned_mesh.get_surface_area()
)

print(
    "\nSurface Area:",
    surface_area
)


edge_manifold = (
    cleaned_mesh.is_edge_manifold()
)

vertex_manifold = (
    cleaned_mesh.is_vertex_manifold()
)

print("\nTopology:")

print(
    "Edge Manifold  :",
    edge_manifold
)

print(
    "Vertex Manifold:",
    vertex_manifold
)


normal_nan = np.isnan(normals).sum()
normal_inf = np.isinf(normals).sum()

print("\nNormal Validation:")

print(
    "NaN:",
    normal_nan
)

print(
    "Infinite:",
    normal_inf
)

print("\n" + "=" * 70)
print("COLOR HANDLING")
print("=" * 70)

print(
    "Uniform mesh coloring is intentionally NOT applied."
)

print(
    "Original RGB data remains available in Phase 2."
)

print(
    "Color/texture transfer will be performed in the "
    "next phase."
)


print("\n" + "=" * 70)
print("FINAL VALIDATION SUMMARY")
print("=" * 70)

validation_pass = (
    final_vertices > 0
    and
    final_triangles > 0
    and
    normal_nan == 0
    and
    normal_inf == 0
)

if validation_pass:

    print(
        "\nFINAL VALIDATION: PASS"
    )

else:

    print(
        "\nFINAL VALIDATION: CHECK REQUIRED"
    )


success = o3d.io.write_triangle_mesh(
    str(OUTPUT_PATH),
    cleaned_mesh,
    write_vertex_normals=True
)

print("\n" + "=" * 70)
print("FINAL OUTPUT")
print("=" * 70)

print("Saved:", success)
print("Output:", OUTPUT_PATH)


print("\nOpening cleaned mesh...")

o3d.visualization.draw_geometries(
    [cleaned_mesh],
    window_name="Cleaned 3D Mesh"
)


print("\n" + "=" * 70)
print("PHASE 4 COMPLETE")
print("=" * 70)