import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D



BASE_DIR = Path(__file__).resolve().parent

POINT_CLOUD_PATH = BASE_DIR.parent / "phase1_depth_estimation" / "outputs" / "point_cloud.npy"
IMAGE_PATH = BASE_DIR.parent / "assest" / "img.jpg"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


points = np.load(POINT_CLOUD_PATH)

print("\n================ POINT CLOUD INFORMATION ================")

print("Points shape:", points.shape)
print("Points dtype:", points.dtype)

print("Points Min:", points.min())
print("Points Max:", points.max())

X = points[:, 0]
Y = points[:, 1]
Z = points[:, 2]

print("\nX Min:", X.min())
print("X Max:", X.max())

print("Y Min:", Y.min())
print("Y Max:", Y.max())

print("Z Min:", Z.min())
print("Z Max:", Z.max())

print("\nNaN values:", np.isnan(points).sum())
print("Infinite values:", np.isinf(points).sum())

print("Total points:", points.shape[0])


z_check = points[:, 2] == 0
zero_depth_count = np.sum(z_check)

print("Zero Depth Points:", zero_depth_count)

print("\n================ IMAGE INFORMATION ================")

img = cv2.imread(str(IMAGE_PATH))

if img is None:
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

print("Image shape:", img.shape)
print("Image dtype:", img.dtype)

image_height, image_width = img.shape[:2]

print("Image height:", image_height)
print("Image width:", image_width)


total_image_pixels = image_height * image_width

print("\nTotal image pixels:", total_image_pixels)
print("Total point-cloud points:", points.shape[0])

if total_image_pixels != points.shape[0]:

    raise ValueError(
        "\nImage pixels and point-cloud points are NOT aligned.\n"
        f"Image pixels: {total_image_pixels}\n"
        f"Point cloud points: {points.shape[0]}"
    )

print("Image and point cloud are correctly aligned.")



cv2.imshow("Original Image", img)



img_rgb = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)



rgb_flat = img_rgb.reshape(-1, 3)

print("\nRGB flat shape:", rgb_flat.shape)
print("RGB dtype:", rgb_flat.dtype)

print("RGB Min:", rgb_flat.min())
print("RGB Max:", rgb_flat.max())



colored_points = np.hstack(
    (
        points,
        rgb_flat
    )
)

print("\n================ COLORED POINT CLOUD ================")

print("Colored points shape:", colored_points.shape)

print("First colored point:")
print(colored_points[0])

print("\nFirst XYZ:")
print(points[0])

print("\nFirst RGB:")
print(rgb_flat[0])


z_colored_points = colored_points[:, 2] == 0

zero_depth_colored = np.sum(
    z_colored_points
)

print(
    "\nZero Depth Points in colored cloud:",
    zero_depth_colored
)


#
# Keep points only when:
# 1. Z > 0
# 2. No NaN
# 3. No Infinite value
#

valid_depth = Z > 0

finite_points = np.all(
    np.isfinite(points),
    axis=1
)

valid_mask = (
    valid_depth
    &
    finite_points
)


count_valid = np.sum(valid_mask)

print("\n================ FILTERING ================")

print("Valid points:", count_valid)

print("Original points:", colored_points.shape[0])

print("Filtered points:", count_valid)

print(
    "Removed points:",
    colored_points.shape[0] - count_valid
)



filtered_points = colored_points[
    valid_mask
]


print(
    "\nFiltered point cloud shape:",
    filtered_points.shape
)


x = filtered_points[:, 0]
y = filtered_points[:, 1]
z = filtered_points[:, 2]



print("\n================ FILTERED XYZ STATISTICS ================")

print("X Min:", x.min())
print("X Max:", x.max())

print("Y Min:", y.min())
print("Y Max:", y.max())

print("Z Min:", z.min())
print("Z Max:", z.max())


x_extent = x.max() - x.min()
y_extent = y.max() - y.min()
z_extent = z.max() - z.min()


print("\n================ XYZ EXTENTS ================")

print("X extent:", x_extent)
print("Y extent:", y_extent)
print("Z extent:", z_extent)



if y_extent != 0:
    xy_ratio = x_extent / y_extent
    zy_ratio = z_extent / y_extent
else:
    xy_ratio = np.nan
    zy_ratio = np.nan


if z_extent != 0:
    xz_ratio = x_extent / z_extent
else:
    xz_ratio = np.nan


print("\nX/Y ratio:", xy_ratio)
print("Z/Y ratio:", zy_ratio)
print("X/Z ratio:", xz_ratio)

x_center = np.mean(x)
y_center = np.mean(y)
z_center = np.mean(z)


print("\n================ CENTROID ================")

print("X centroid:", x_center)
print("Y centroid:", y_center)
print("Z centroid:", z_center)



rgb_colors = filtered_points[:, 3:6]

print("\n================ RGB INFORMATION ================")

print("RGB shape:", rgb_colors.shape)
print("RGB dtype:", rgb_colors.dtype)

print("RGB Min:", rgb_colors.min())
print("RGB Max:", rgb_colors.max())


normalized_colors = (
    rgb_colors.astype(np.float32) / 255.0
)


print("\nNormalized RGB shape:", normalized_colors.shape)

print("Normalized RGB Min:", normalized_colors.min())
print("Normalized RGB Max:", normalized_colors.max())



print("\n================ 3D VISUALIZATION ================")

fig = plt.figure(
    figsize=(10, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


ax.scatter(
    filtered_points[:, 0],
    filtered_points[:, 1],
    filtered_points[:, 2],
    s=0.2,
    c=normalized_colors
)


ax.set_box_aspect(
    (
        x_extent,
        y_extent,
        z_extent
    )
)


ax.view_init(
    elev=25,
    azim=45
)


ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title(
    "Filtered Colored 3D Point Cloud"
)


plt.tight_layout()
plt.show()
print("\n================ FINAL RESULTS ================")

print(
    "Final point cloud shape:",
    filtered_points.shape
)

print(
    "Final number of points:",
    filtered_points.shape[0]
)

print("X Min:", x.min())
print("X Max:", x.max())

print("Y Min:", y.min())
print("Y Max:", y.max())

print("Z Min:", z.min())
print("Z Max:", z.max())

print("RGB Min:", rgb_colors.min())
print("RGB Max:", rgb_colors.max())

print(
    "Normalized RGB Min:",
    normalized_colors.min()
)

print(
    "Normalized RGB Max:",
    normalized_colors.max()
)

print(
    "NaN values in original points:",
    np.isnan(points).sum()
)

print(
    "Infinite values in original points:",
    np.isinf(points).sum()
)


output_file = (
    OUTPUT_DIR /
    "filtered_point_cloud.npy"
)

np.save(
    output_file,
    filtered_points
)


# Also save XYZ and RGB separately
np.save(
    OUTPUT_DIR / "filtered_xyz.npy",
    filtered_points[:, :3]
)

np.save(
    OUTPUT_DIR / "filtered_rgb.npy",
    filtered_points[:, 3:6]
)


print("\n================ SAVED OUTPUTS ================")

print(
    "Filtered point cloud:",
    output_file
)

print(
    "Filtered XYZ:",
    OUTPUT_DIR / "filtered_xyz.npy"
)

print(
    "Filtered RGB:",
    OUTPUT_DIR / "filtered_rgb.npy"
)

cv2.waitKey(0)
cv2.destroyAllWindows()