import cv2
import numpy as np
import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#==========import model==============================
sibling_dir = Path(__file__).resolve().parent.parent / "Depth-Anything-V2"
sys.path.append(str(sibling_dir))

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

print("Device:", DEVICE)


#============model configuration========================
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
}

#================model load=============================
encoder = 'vits'

model = DepthAnythingV2(**model_configs[encoder])

model.load_state_dict(
    torch.load(
        '../models/depth_anything_v2_vits.pth',
        map_location='cpu'
    )
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully")
#================load image===============================
img=cv2.imread('../assest/img.jpg')


#===================image inspection======================
print(img.dtype)
print(img.size)
print(img.min())
print(img.max())
print(img[0,255])

#========================image preprocession===================
avg_img=np.mean(img,axis=(0,1))
avg_all=np.mean(avg_img)
std=np.std(img,axis=(0,1))
print("Average brightness: ",avg_all)
print("Standard Deviation: ",std)

img_process=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


print(img[0,255])
print(img_process[0,255])

img_resize=cv2.resize(img,(306,204),interpolation=cv2.INTER_LINEAR)
print(img_resize.shape)

img_upscale=cv2.resize(img,(612,408),interpolation=cv2.INTER_LINEAR)
print(img_upscale.shape)


#==========================normalize===========================
normalized_img=img.astype(np.float32)/255.0
print("Original dtype: ",img.dtype)
print("Normalized dtype: ",normalized_img.dtype)
print("Original MIN: ",img.min())
print("Normalized MIN: ",normalized_img.min())
print("Original MAX: ",img.max())
print("Normalized MAX: ",normalized_img.max())

print("Original pixel:", img[0, 255])
print("Normalized pixel:", normalized_img[0, 255])
cv2.imshow('img',img)
cv2.imshow('img upscale',img_upscale)

cv2.imshow('img resize',img_resize)
cv2.imshow('Normalized image',normalized_img)

print("Original image",img.shape)
print("Downscaled image",img_resize.shape)
print("Upscaled image",img_upscale.shape)
print("Normalized image",normalized_img.shape)

print("Original image",img.dtype)
print("Downscaled image",img_resize.dtype)
print("Upscaled image",img_upscale.dtype)
print("Normalized image",normalized_img.dtype)

#==========================depth estimation image======================
depth=model.infer_image(img)
print("Depth Min:", depth.min())
print("Depth Max:", depth.max())
print("Depth Mean:", depth.mean())
print("Depth Median:", np.median(depth))

print("Top-left:", depth[50, 50])
print("Center:", depth[204, 306])
print("Bottom-center:", depth[350, 306])
print("Depth Dtype",depth.dtype)
print("Depth Size",depth.size)
print("Depth Shape", depth.shape)


#==========================normalize depth===========================
normalized_depth=(depth-depth.min())/(depth.max()-depth.min())
normalized_depth=(normalized_depth*255).astype(np.uint8)
cv2.imshow("normalize depth",normalized_depth)

print("Top-left depth:", depth[50, 50])
print("Center depth:", depth[204, 306])
print("Bottom-center depth:", depth[350, 306])

#==========================depth statistics==============================

print("Depth Min",depth.min())
print("Depth Max",depth.max())
avg_depth=np.mean(depth,axis=(0,1))
depth_median=np.median(depth,axis=(0,1))
depth_std=np.std(depth,axis=(0,1))
print("Depth Mean",avg_depth)
print("Depth Median",depth_median)
print("Depth Standard Deviation",depth_std)

#=======================depth distribution===========================
depth_flatten=depth.flatten()
plt.hist(depth_flatten,bins=30, edgecolor='black')

plt.title('Histrogram')
plt.xlabel('Depth Value')
plt.ylabel('Number of Pixels')
plt.show()

bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

counts, edges = np.histogram(depth_flatten, bins=bins)

for i in range(len(counts)):
    print(
        f"{edges[i]:.0f}-{edges[i+1]:.0f}: {counts[i]} pixels"
    )

depth_grad_y,depth_grad_x=np.gradient(depth)
gradient_magnitude = np.sqrt(
    depth_grad_x**2 + depth_grad_y**2
)
print("Gradient Min:", gradient_magnitude.min())
print("Gradient Max:", gradient_magnitude.max())
print("Gradient Mean:", gradient_magnitude.mean())
print("Gradient Std:", gradient_magnitude.std())
print("Gradient Shape:", gradient_magnitude.shape)
gradient_norm = cv2.normalize(
    gradient_magnitude,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)
cv2.imshow("Depth Gradient", gradient_norm)
cv2.imwrite("outputs/depth_gradient.png", gradient_norm)
threshold = gradient_magnitude.mean() + gradient_magnitude.std()
print("Threshold:", threshold)
high_gradient = gradient_magnitude > threshold
high_gradient_count = np.sum(high_gradient)
print("High Gradient Pixels:", high_gradient_count)
top_region = depth[0:136, :]
middle_region = depth[136:272, :]
bottom_region = depth[272:408, :]
print("Top:", top_region.shape)
print("Middle:", middle_region.shape)
print("Bottom:", bottom_region.shape)

#============================depth statistics========================

top_depth=np.mean(top_region,axis=(0,1))
top_median=np.median(top_region,axis=(0,1))
top_std=np.std(top_region,axis=(0,1))
print("Depth top Mean",top_depth)
print("Depth top Median",top_median)
print("Depth top Standard Deviation",top_std)


middle_depth=np.mean(middle_region,axis=(0,1))
middle_median=np.median(middle_region,axis=(0,1))
middle_std=np.std(middle_region,axis=(0,1))
print("Depth middle Mean",middle_depth)
print("Depth middle Median",middle_median)
print("Depth middle Standard Deviation",middle_std)


bottom_depth=np.mean(bottom_region,axis=(0,1))
bottom_median=np.median(bottom_region,axis=(0,1))
bottom_std=np.std(bottom_region,axis=(0,1))
print("Depth bottom Mean",bottom_depth)
print("Depth bottom Median",bottom_median)
print("Depth bottom Standard Deviation",bottom_std)

top_middle_change = middle_depth - top_depth
middle_bottom_change = bottom_depth - middle_depth

print("Top to Middle Depth Change:", top_middle_change)
print("Middle to Bottom Depth Change:", middle_bottom_change)
top_bottom_change = bottom_depth - top_depth
print("Top to Bottom Depth Change:", top_bottom_change)


#===========================camera=======================
cx = 612 / 2
cy = 408 / 2

print("Principal Point X:", cx)
print("Principal Point Y:", cy)

fx = 612
fy = 612

print("Focal Length X:", fx)
print("Focal Length Y:", fy)

u = 400
v = 250

print("Pixel X:", u)
print("Pixel Y:", v)

x = (u - cx) / fx
y = (v - cy) / fy

print("Camera X:", x)
print("Camera Y:", y)

k=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
print("Camera Matrix K:")
print(k)
print("K shape",k.shape)
print("K dtype",k.dtype)

K_inv = np.linalg.inv(k)

print("K inverse:")
print(K_inv)
print("K inverse shape:", K_inv.shape)
pixel = np.array([400, 250, 1])
camera_point = K_inv @ pixel
print("Camera coordinate:", camera_point)

Z = depth[v, u]

print("Pixel U:", u)
print("Pixel V:", v)
print("Depth Z:", Z)

X=(u-cx)*Z/fx
Y=(v-cy)*Z/fy

print("3D X:", X)
print("3D Y:", Y)
print("3D Z:", Z)

u_coords, v_coords = np.meshgrid(
    np.arange(img.shape[1]),
    np.arange(img.shape[0])
)

print("U shape:", u_coords.shape)
print("V shape:", v_coords.shape)
X = ((u_coords - cx) * depth) / fx
Y = ((v_coords - cy) * depth) / fy
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Z shape:", depth.shape)
points_3d = np.stack((X, Y, depth), axis=-1)
print("3D Points Shape:", points_3d.shape)
points_flat = points_3d.reshape(-1, 3)
print("Flattened 3D Points Shape:", points_flat.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    points_flat[:, 0],
    points_flat[:, 1],
    points_flat[:, 2],
    s=1
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
np.save(
    "outputs/point_cloud.npy",
    points_flat
)
cv2.waitKey(0)
cv2.destroyAllWindows()