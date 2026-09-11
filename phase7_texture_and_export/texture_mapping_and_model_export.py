import numpy as np
from pathlib import Path
from PIL import Image
import trimesh

BASE_DIR = Path(__file__).resolve().parent

MESH_PATH = (
    BASE_DIR.parent
    / "phase5_mesh_refinement"
    / "outputs"
    / "refined_colored_mesh.ply"
)

# Original image used in Phase 1 / Phase 2
IMAGE_PATH = (
    BASE_DIR.parent
    / "assest"
    / "img.jpg"
)

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OBJ_PATH = OUTPUT_DIR / "final_3d_model.obj"
GLB_PATH = OUTPUT_DIR / "final_3d_model.glb"
TEXTURE_PATH = OUTPUT_DIR / "final_texture.png"

FX = 612.0
FY = 612.0
CX = 306.0
CY = 204.0

print("=" * 70)
print("PHASE 7 - TEXTURE MAPPING + FINAL EXPORT")
print("=" * 70)

mesh_o3d = None

# Open3D is used only for reliable PLY reading
import open3d as o3d

mesh_o3d = o3d.io.read_triangle_mesh(
    str(MESH_PATH)
)

if mesh_o3d.is_empty():
    raise ValueError(
        "Phase 5 mesh is empty."
    )

vertices = np.asarray(
    mesh_o3d.vertices,
    dtype=np.float64
)

faces = np.asarray(
    mesh_o3d.triangles,
    dtype=np.int64
)

print("\nInput mesh:")
print("Vertices :", len(vertices))
print("Triangles:", len(faces))


# ============================================================
# VALIDATE MESH
# ============================================================

if len(vertices) == 0:
    raise ValueError("No mesh vertices found.")

if len(faces) == 0:
    raise ValueError("No mesh triangles found.")

if (
    np.isnan(vertices).any()
    or
    np.isinf(vertices).any()
):
    raise ValueError(
        "Mesh contains NaN/Inf vertex positions."
    )

if (
    np.min(faces) < 0
    or
    np.max(faces) >= len(vertices)
):
    raise ValueError(
        "Mesh contains invalid triangle indices."
    )


print("\nLoading original image...")

image = Image.open(
    IMAGE_PATH
).convert("RGB")

image_width, image_height = image.size

print(
    "Image width :",
    image_width
)

print(
    "Image height:",
    image_height
)

print("\nCamera intrinsics:")
print("fx:", FX)
print("fy:", FY)
print("cx:", CX)
print("cy:", CY)

if (
    CX <= 0
    or CY <= 0
    or CX >= image_width
    or CY >= image_height
):
    print(
        "\nWARNING: Principal point is outside the image."
    )

if (
    abs(CX - image_width / 2.0) > 5
    or
    abs(CY - image_height / 2.0) > 5
):
    print(
        "\nWARNING: Principal point is not near image center."
    )
    print(
        "Make sure these values match Phase 1."
    )


# Perspective projection:
#
# u = fx * X / Z + cx
# v = fy * Y / Z + cy
#
# Image UV coordinates:
#
# U = u / (width - 1)
# V = 1 - v / (height - 1)
#
# V is flipped because image coordinates start at
# the top-left while UV coordinates start at bottom-left.
#

print("\nProjecting mesh vertices onto image...")

X = vertices[:, 0]
Y = vertices[:, 1]
Z = vertices[:, 2]


valid_depth = Z > 1e-8

if not np.all(valid_depth):

    invalid_count = np.sum(
        ~valid_depth
    )

    print(
        "WARNING:",
        invalid_count,
        "vertices have non-positive depth."
    )

    # Prevent division by zero
    safe_Z = np.where(
        valid_depth,
        Z,
        1e-8
    )

else:

    safe_Z = Z


u = (
    FX * X / safe_Z
    +
    CX
)

v = (
    FY * Y / safe_Z
    +
    CY
)


# ============================================================
# UV COORDINATES
# ============================================================

uv_u = (
    u / (image_width - 1)
)

uv_v = (
    1.0
    -
    v / (image_height - 1)
)

uv = np.column_stack(
    [
        uv_u,
        uv_v
    ]
)

outside_u = (
    (uv[:, 0] < 0.0)
    |
    (uv[:, 0] > 1.0)
)

outside_v = (
    (uv[:, 1] < 0.0)
    |
    (uv[:, 1] > 1.0)
)

outside_uv = (
    outside_u
    |
    outside_v
    |
    (~valid_depth)
)

print("\nUV projection statistics:")

print(
    "Vertices outside image:",
    np.sum(outside_uv)
)

print(
    "Percentage outside:",
    np.mean(outside_uv) * 100
)


# Clamp UV values so texture lookup stays inside image
uv = np.clip(
    uv,
    0.0,
    1.0
)

print("\nSaving texture...")

image.save(
    TEXTURE_PATH
)

print(
    "Texture:",
    TEXTURE_PATH
)



print("\nCreating textured mesh...")

triangle_mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    process=False
)

material = trimesh.visual.material.SimpleMaterial(
    image=image
)

visual = trimesh.visual.texture.TextureVisuals(
    uv=uv,
    image=image,
    material=material
)

triangle_mesh.visual = visual

print("\n" + "=" * 70)
print("7.1 - FINAL MESH VALIDATION")
print("=" * 70)

print(
    "Vertices:",
    len(triangle_mesh.vertices)
)

print(
    "Triangles:",
    len(triangle_mesh.faces)
)

print(
    "Watertight:",
    triangle_mesh.is_watertight
)

print(
    "Convex:",
    triangle_mesh.is_convex
)

print(
    "Euler Number:",
    triangle_mesh.euler_number
)

print(
    "Surface Area:",
    triangle_mesh.area
)

print(
    "Has Visual:",
    triangle_mesh.visual is not None
)

print(
    "Has UV:",
    triangle_mesh.visual.uv is not None
)

if triangle_mesh.visual.uv is not None:

    print(
        "UV Shape:",
        triangle_mesh.visual.uv.shape
    )

    print(
        "UV Min:",
        np.min(triangle_mesh.visual.uv)
    )

    print(
        "UV Max:",
        np.max(triangle_mesh.visual.uv)
    )


print("\n" + "=" * 70)
print("7.2 - OBJ EXPORT")
print("=" * 70)

obj_result = triangle_mesh.export(
    file_obj=str(OBJ_PATH),
    file_type="obj",
    include_texture=True
)

print(
    "OBJ export complete."
)

print(
    "OBJ path:",
    OBJ_PATH
)


print("\n" + "=" * 70)
print("7.3 - GLB EXPORT")
print("=" * 70)

glb_result = triangle_mesh.export(
    file_obj=str(GLB_PATH),
    file_type="glb"
)

print(
    "GLB export complete."
)

print(
    "GLB path:",
    GLB_PATH
)



print("\n" + "=" * 70)
print("7.4 - OUTPUT CHECK")
print("=" * 70)

files_to_check = [
    OBJ_PATH,
    GLB_PATH,
    TEXTURE_PATH
]

all_files_exist = True

for file_path in files_to_check:

    exists = file_path.exists()

    print(
        file_path.name,
        ":",
        exists
    )

    if exists:

        print(
            "  Size:",
            file_path.stat().st_size,
            "bytes"
        )

    else:

        all_files_exist = False


print("\n" + "=" * 70)
print("PHASE 7 FINAL SUMMARY")
print("=" * 70)

print(
    "Input mesh:",
    MESH_PATH
)

print(
    "Original texture:",
    IMAGE_PATH
)

print(
    "Final OBJ:",
    OBJ_PATH
)

print(
    "Final GLB:",
    GLB_PATH
)

print(
    "Final Texture:",
    TEXTURE_PATH
)

if all_files_exist:

    print(
        "\nPHASE 7 VALIDATION: PASS"
    )

else:

    print(
        "\nPHASE 7 VALIDATION: FAILED"
    )


print("\n" + "=" * 70)
print("FINAL 3D MODEL EXPORT COMPLETE")
print("=" * 70)
