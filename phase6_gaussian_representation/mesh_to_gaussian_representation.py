import open3d as o3d
import numpy as np
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent

MESH_PATH = (
    BASE_DIR.parent
    / "phase5"
    / "outputs"
    / "refined_colored_mesh.ply"
)

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "gaussian_representation_final.npz"
)


# ============================================================
# LOAD COLORED MESH
# ============================================================

mesh = o3d.io.read_triangle_mesh(
    str(MESH_PATH)
)

if mesh.is_empty():
    raise ValueError(
        "Phase 5 colored mesh is empty."
    )


print("=" * 70)
print("PHASE 6 - GAUSSIAN REPRESENTATION BASELINE")
print("=" * 70)


# ============================================================
# BASIC MESH INFORMATION
# ============================================================

vertices = np.asarray(
    mesh.vertices,
    dtype=np.float64
)

triangles = np.asarray(
    mesh.triangles
)

print("\nMesh:")
print(mesh)

print(
    "\nVertices:",
    len(vertices)
)

print(
    "Triangles:",
    len(triangles)
)

print(
    "Vertex shape:",
    vertices.shape
)

print(
    "Vertex dtype:",
    vertices.dtype
)


# ============================================================
# POSITION VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("6.1 - POSITION VALIDATION")
print("=" * 70)

print(
    "Position NaN:",
    np.isnan(vertices).sum()
)

print(
    "Position Inf:",
    np.isinf(vertices).sum()
)


if (
    np.isnan(vertices).any()
    or
    np.isinf(vertices).any()
):

    raise ValueError(
        "Invalid vertex positions detected."
    )


# ============================================================
# BOUNDING BOX
# ============================================================

print("\nBounding Box:")

print(
    "X Min:",
    np.min(vertices[:, 0])
)

print(
    "X Max:",
    np.max(vertices[:, 0])
)

print(
    "Y Min:",
    np.min(vertices[:, 1])
)

print(
    "Y Max:",
    np.max(vertices[:, 1])
)

print(
    "Z Min:",
    np.min(vertices[:, 2])
)

print(
    "Z Max:",
    np.max(vertices[:, 2])
)


# ============================================================
# MESH STATISTICS
# ============================================================

print("\nSurface Area:")
print(
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
# GAUSSIAN POSITIONS
# ============================================================

gaussian_positions = vertices.copy()

gaussian_count = (
    len(gaussian_positions)
)

print("\n" + "=" * 70)
print("6.2 - GAUSSIAN POSITIONS")
print("=" * 70)

print(
    "Gaussian Count:",
    gaussian_count
)

print(
    "Gaussian Position Shape:",
    gaussian_positions.shape
)


# ============================================================
# CREATE POINT CLOUD
# ============================================================

pcd = o3d.geometry.PointCloud()

pcd.points = (
    o3d.utility.Vector3dVector(
        gaussian_positions
    )
)


# ============================================================
# NEAREST-NEIGHBOR DISTANCES
# ============================================================

print("\n" + "=" * 70)
print("6.3 - NEAREST NEIGHBOR ANALYSIS")
print("=" * 70)

distances = (
    pcd.compute_nearest_neighbor_distance()
)

distances = np.asarray(
    distances,
    dtype=np.float64
)


# Remove invalid distances
valid_distance_mask = (
    np.isfinite(distances)
    &
    (distances > 0)
)

valid_distances = (
    distances[valid_distance_mask]
)


if len(valid_distances) == 0:

    raise ValueError(
        "No valid nearest-neighbor distances found."
    )


print(
    "Mean distance:",
    np.mean(valid_distances)
)

print(
    "Median distance:",
    np.median(valid_distances)
)

print(
    "Min distance:",
    np.min(valid_distances)
)

print(
    "Max distance:",
    np.max(valid_distances)
)


# ============================================================
# VERY CLOSE POINTS
# ============================================================

very_close_mask = (
    distances < 1e-6
)

very_close_count = (
    np.sum(very_close_mask)
)

print(
    "\nVery close points:",
    very_close_count
)

print(
    "Percentage:",
    very_close_count
    /
    len(distances)
    *
    100
)


# ============================================================
# GAUSSIAN SCALES
# ============================================================
#
# We use local point spacing to estimate Gaussian size.
#
# Instead of a single scalar scale, we create
# isotropic XYZ scales here.
#
# Each Gaussian gets:
# [scale, scale, scale]
#

base_scale = (
    distances * 0.5
)


# Replace invalid / zero values
median_scale = np.median(
    valid_distances
) * 0.5

minimum_scale = (
    median_scale * 0.01
)

if minimum_scale <= 0:
    minimum_scale = 1e-6


base_scale = np.where(
    np.isfinite(base_scale)
    &
    (base_scale > 0),
    base_scale,
    minimum_scale
)


base_scale = np.maximum(
    base_scale,
    minimum_scale
)


# ============================================================
# SCALE OUTLIER ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("6.4 - GAUSSIAN SCALE ANALYSIS")
print("=" * 70)

print(
    "Scale Mean:",
    np.mean(base_scale)
)

print(
    "Scale Median:",
    np.median(base_scale)
)

print(
    "Scale Min:",
    np.min(base_scale)
)

print(
    "Scale Max:",
    np.max(base_scale)
)

print(
    "Scale Std:",
    np.std(base_scale)
)

scale_95 = np.percentile(
    base_scale,
    95
)

scale_99 = np.percentile(
    base_scale,
    99
)

print(
    "95th Percentile:",
    scale_95
)

print(
    "99th Percentile:",
    scale_99
)


# ============================================================
# CAP EXTREME SCALES
# ============================================================

scale_outlier_mask = (
    base_scale > scale_99
)

scale_outlier_count = (
    np.sum(scale_outlier_mask)
)

print(
    "\nScale Outlier Count:",
    scale_outlier_count
)

print(
    "Scale Outlier Percentage:",
    scale_outlier_count
    /
    len(base_scale)
    *
    100
)


# Cap extreme values
base_scale = np.minimum(
    base_scale,
    scale_99
)


# ============================================================
# CREATE XYZ SCALES
# ============================================================

gaussian_scales = np.column_stack(
    [
        base_scale,
        base_scale,
        base_scale
    ]
)

print(
    "\nGaussian Scales Shape:",
    gaussian_scales.shape
)


# ============================================================
# SCALE VALIDATION
# ============================================================

print(
    "Scale NaN:",
    np.isnan(gaussian_scales).sum()
)

print(
    "Scale Inf:",
    np.isinf(gaussian_scales).sum()
)

print(
    "Scale <= 0:",
    np.sum(gaussian_scales <= 0)
)


# ============================================================
# COMPUTE VERTEX NORMALS
# ============================================================

if not mesh.has_vertex_normals():

    mesh.compute_vertex_normals()


normals = np.asarray(
    mesh.vertex_normals,
    dtype=np.float64
)


# ============================================================
# NORMAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("6.5 - NORMAL VALIDATION")
print("=" * 70)

print(
    "Normal Shape:",
    normals.shape
)

print(
    "Normal NaN:",
    np.isnan(normals).sum()
)

print(
    "Normal Inf:",
    np.isinf(normals).sum()
)


# Normalize normals
normal_lengths = np.linalg.norm(
    normals,
    axis=1,
    keepdims=True
)

valid_normals = (
    np.isfinite(normal_lengths[:, 0])
    &
    (normal_lengths[:, 0] > 1e-12)
)

normals[valid_normals] /= (
    normal_lengths[valid_normals]
)


# ============================================================
# GAUSSIAN ROTATIONS
# ============================================================
#
# Quaternion format:
# [w, x, y, z]
#
# Rotation maps +Z direction toward
# the corresponding surface normal.
#

z_axis = np.array(
    [0.0, 0.0, 1.0]
)


dots = np.sum(
    normals * z_axis,
    axis=1
)

dots = np.clip(
    dots,
    -1.0,
    1.0
)


angles = np.arccos(
    dots
)

half_angles = (
    angles / 2.0
)


axes = np.cross(
    np.tile(
        z_axis,
        (len(normals), 1)
    ),
    normals
)

axis_norms = np.linalg.norm(
    axes,
    axis=1
)


# Initialize rotations
gaussian_rotations = np.zeros(
    (len(normals), 4),
    dtype=np.float64
)


# Normal case
normal_axis_mask = (
    axis_norms > 1e-12
)

axes[normal_axis_mask] /= (
    axis_norms[
        normal_axis_mask
    ][:, None]
)


sin_half = np.sin(
    half_angles
)


gaussian_rotations[:, 0] = (
    np.cos(half_angles)
)

gaussian_rotations[:, 1:] = (
    axes
    *
    sin_half[:, None]
)


# ============================================================
# SPECIAL CASE: NORMAL = -Z
# ============================================================

negative_z_mask = (
    dots < -1.0 + 1e-10
)

gaussian_rotations[
    negative_z_mask
] = np.array(
    [0.0, 1.0, 0.0, 0.0]
)


# ============================================================
# SPECIAL CASE: NORMAL = +Z
# ============================================================

positive_z_mask = (
    dots > 1.0 - 1e-10
)

gaussian_rotations[
    positive_z_mask
] = np.array(
    [1.0, 0.0, 0.0, 0.0]
)


# ============================================================
# ROTATION VALIDATION
# ============================================================

rotation_norms = np.linalg.norm(
    gaussian_rotations,
    axis=1
)

print("\n" + "=" * 70)
print("6.6 - ROTATION VALIDATION")
print("=" * 70)

print(
    "Rotation Shape:",
    gaussian_rotations.shape
)

print(
    "Rotation NaN:",
    np.isnan(gaussian_rotations).sum()
)

print(
    "Rotation Inf:",
    np.isinf(gaussian_rotations).sum()
)

print(
    "Quaternion Norm Min:",
    np.min(rotation_norms)
)

print(
    "Quaternion Norm Max:",
    np.max(rotation_norms)
)

print(
    "Quaternion Norm Mean:",
    np.mean(rotation_norms)
)


# ============================================================
# GAUSSIAN COLORS
# ============================================================

print("\n" + "=" * 70)
print("6.7 - GAUSSIAN COLORS")
print("=" * 70)

if not mesh.has_vertex_colors():

    raise ValueError(
        "Phase 5 mesh does not contain vertex colors."
    )


gaussian_colors = np.asarray(
    mesh.vertex_colors,
    dtype=np.float64
)


print(
    "Color Shape:",
    gaussian_colors.shape
)

print(
    "Color Min:",
    np.min(gaussian_colors)
)

print(
    "Color Max:",
    np.max(gaussian_colors)
)

print(
    "Color NaN:",
    np.isnan(gaussian_colors).sum()
)

print(
    "Color Inf:",
    np.isinf(gaussian_colors).sum()
)


# ============================================================
# ENSURE RGB RANGE
# ============================================================

gaussian_colors = np.clip(
    gaussian_colors,
    0.0,
    1.0
)


# ============================================================
# GAUSSIAN OPACITY
# ============================================================
#
# This is a baseline constant opacity.
# It is NOT optimized opacity.
#

gaussian_opacities = np.full(
    gaussian_count,
    0.5,
    dtype=np.float64
)


print("\n" + "=" * 70)
print("6.8 - OPACITY")
print("=" * 70)

print(
    "Opacity Shape:",
    gaussian_opacities.shape
)

print(
    "Opacity Min:",
    np.min(gaussian_opacities)
)

print(
    "Opacity Max:",
    np.max(gaussian_opacities)
)

print(
    "Opacity Mean:",
    np.mean(gaussian_opacities)
)

print(
    "Unique Opacity:",
    np.unique(gaussian_opacities)
)



print("\n" + "=" * 70)
print("6.9 - SPATIAL OUTLIER ANALYSIS")
print("=" * 70)

centroid = np.mean(
    gaussian_positions,
    axis=0
)

distance_from_centroid = (
    np.linalg.norm(
        gaussian_positions
        -
        centroid,
        axis=1
    )
)

distance_mean = np.mean(
    distance_from_centroid
)

distance_std = np.std(
    distance_from_centroid
)

distance_threshold = (
    distance_mean
    +
    3.0
    *
    distance_std
)

spatial_outlier_mask = (
    distance_from_centroid
    >
    distance_threshold
)


print(
    "Centroid:",
    centroid
)

print(
    "Distance Mean:",
    distance_mean
)

print(
    "Distance Std:",
    distance_std
)

print(
    "Outlier Threshold:",
    distance_threshold
)

print(
    "Spatial Outliers:",
    np.sum(spatial_outlier_mask)
)

print(
    "Spatial Outlier Percentage:",
    np.mean(
        spatial_outlier_mask
    )
    *
    100
)


scale_outlier_mask = (
    base_scale > scale_99
)

both_outlier_mask = (
    scale_outlier_mask
    &
    spatial_outlier_mask
)

print(
    "Both Scale + Spatial Outliers:",
    np.sum(both_outlier_mask)
)

print("\n" + "=" * 70)
print("6.10 - FINAL GAUSSIAN VALIDATION")
print("=" * 70)


print(
    "Positions:",
    len(gaussian_positions)
)

print(
    "Scales:",
    len(gaussian_scales)
)

print(
    "Rotations:",
    len(gaussian_rotations)
)

print(
    "Opacities:",
    len(gaussian_opacities)
)

print(
    "Colors:",
    len(gaussian_colors)
)


# ------------------------------------------------------------
# Shape consistency
# ------------------------------------------------------------

same_count = (
    len(gaussian_positions)
    ==
    len(gaussian_scales)
    ==
    len(gaussian_rotations)
    ==
    len(gaussian_opacities)
    ==
    len(gaussian_colors)
)

print(
    "\nAll Gaussian arrays same count:",
    same_count
)


position_invalid = (
    np.isnan(gaussian_positions).any()
    or
    np.isinf(gaussian_positions).any()
)

scale_invalid = (
    np.isnan(gaussian_scales).any()
    or
    np.isinf(gaussian_scales).any()
    or
    np.any(gaussian_scales <= 0)
)

rotation_invalid = (
    np.isnan(gaussian_rotations).any()
    or
    np.isinf(gaussian_rotations).any()
)

opacity_invalid = (
    np.isnan(gaussian_opacities).any()
    or
    np.isinf(gaussian_opacities).any()
)

color_invalid = (
    np.isnan(gaussian_colors).any()
    or
    np.isinf(gaussian_colors).any()
)


quaternion_invalid = (
    np.any(
        np.abs(
            rotation_norms - 1.0
        )
        >
        1e-5
    )
)


print(
    "Positions valid:",
    not position_invalid
)

print(
    "Scales valid:",
    not scale_invalid
)

print(
    "Rotations valid:",
    not rotation_invalid
)

print(
    "Opacities valid:",
    not opacity_invalid
)

print(
    "Colors valid:",
    not color_invalid
)

print(
    "Quaternion norms valid:",
    not quaternion_invalid
)



final_validation = (
    same_count
    and
    not position_invalid
    and
    not scale_invalid
    and
    not rotation_invalid
    and
    not opacity_invalid
    and
    not color_invalid
    and
    not quaternion_invalid
)


print("\n" + "=" * 70)

if final_validation:

    print(
        "FINAL VALIDATION: PASS"
    )

else:

    print(
        "FINAL VALIDATION: FAILED"
    )
    raise ValueError(
        "Gaussian representation validation failed."
    )



np.savez(
    str(OUTPUT_PATH),
    positions=gaussian_positions,
    scales=gaussian_scales,
    rotations=gaussian_rotations,
    opacities=gaussian_opacities,
    colors=gaussian_colors
)


print("\n" + "=" * 70)
print("6.11 - FINAL OUTPUT")
print("=" * 70)

print(
    "Saved:",
    OUTPUT_PATH
)

print(
    "Gaussian Count:",
    gaussian_count
)

print(
    "Position Shape:",
    gaussian_positions.shape
)

print(
    "Scale Shape:",
    gaussian_scales.shape
)

print(
    "Rotation Shape:",
    gaussian_rotations.shape
)

print(
    "Opacity Shape:",
    gaussian_opacities.shape
)

print(
    "Color Shape:",
    gaussian_colors.shape
)

print("\n" + "=" * 70)
print("PHASE 6 COMPLETE")
print("=" * 70)