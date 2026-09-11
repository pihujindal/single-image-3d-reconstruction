import numpy as np
import cv2
import open3d as o3d

#===================file load=================================
filtered_points=np.load("../phase2/outputs/filtered_point_cloud.npy")
print("Shape: ",filtered_points.shape)
print("DType: ",filtered_points.dtype)

xyz_points=filtered_points[:,:3]
print("X Y Z shape: ",xyz_points.shape)
print("X Y Z DType: ",xyz_points.dtype)

#===================check nan and infinite value===================
print("NAN Values: ",np.isnan(xyz_points).sum())
print("Infinite Values: ",np.isinf(xyz_points).sum())

#==================open 3d point cloud============================
point_cloud=o3d.geometry.PointCloud()
point_cloud.points=o3d.utility.Vector3dVector(xyz_points)
print(point_cloud)

search_param = o3d.geometry.KDTreeSearchParamKNN(knn=30)

point_cloud.estimate_normals(
    search_param=search_param
)
normals = np.asarray(point_cloud.normals)

print("Normals shape:", normals.shape)
print("Normals count:", len(normals))
print("Normals nan:", np.isnan(normals).sum())
print("Normals infinite:", np.isinf(normals).sum())
normal_magnitude = np.linalg.norm(normals, axis=1)
print("Normals magnitude min",normal_magnitude.min())
print("Normals magnitude max",normal_magnitude.max())
print("Normals magnitude mean",normal_magnitude.mean())

#======================mesh reconstruction======================
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    point_cloud,
    depth=9
)

print(mesh)
print("Densities:", len(densities))
np.save("densities.npy", np.asarray(densities))
vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)
print("vertices dtype: ",vertices.dtype)
print("vertices shape: ",vertices.shape)
print("triangles dtype: ",triangles.dtype)
print("triangles shape: ",triangles.shape)
print("triangle min: ",np.min(triangles))
print("triangle max: ",np.max(triangles))
print("Vertices nan: ",np.isnan(vertices).sum())
print("Vertices Infinite: ",np.isinf(vertices).sum())
print("Mesh vertices",len(mesh.vertices))
print("Mesh triangles",len(mesh.triangles))
vertices_x=vertices[:,0]
vertices_y=vertices[:,1]
vertices_z=vertices[:,2]

print("x Min: ",vertices_x.min())
print("x Max: ",vertices_x.max())
print("y Min: ",vertices_y.min())
print("y Max: ",vertices_y.max())
print("z Min: ",vertices_z.min())
print("z Max: ",vertices_z.max())
print(mesh.get_surface_area())
print(mesh.is_empty())
print(mesh.is_edge_manifold())
print(mesh.is_vertex_manifold())
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
o3d.io.write_triangle_mesh("reconstructed_mesh.ply", mesh)