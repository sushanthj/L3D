"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
import imageio
import time
import sys

from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image


def load_rgbd_data(path="data/rgbd_data.pkl"):
    """
    rgb_data.keys() = dict_keys(['rgb1', 'mask1', 'depth1',
                                'rgb2', 'mask2', 'depth2',
                                'cameras1', 'cameras2'])
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_first_pointcloud(
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color)
    rgb_data = load_rgbd_data()

    points, rgb = unproject_depth_image(rgb_data.get("rgb1"), rgb_data.get("mask1"),
                                        rgb_data.get("depth1"), rgb_data.get("cameras1"))
    # extend points to 12 views
    num_views = 12
    points = points.unsqueeze(0).to(device)
    rgb = rgb.unsqueeze(0).to(device)
    points = points.expand(num_views, -1, -1)
    rgb = rgb.expand(num_views, -1, -1)
    point_cloud = pytorch3d.structures.Pointclouds(points=points, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=6,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )

    # Additional rotation of 180 degrees about the z-axis
    additional_rotation = torch.tensor([[-1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, 1]], dtype=R.dtype, device=R.device)
    additional_rotation = additional_rotation.unsqueeze(0).expand(num_views, -1, -1)

    # Apply additional rotation to the original rotations
    R_additional = R @ additional_rotation

    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_additional,
        T=T,
        device=device
    )

    images = renderer(point_cloud, cameras=many_cameras)
    images = images.cpu().numpy()
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255)[:,:,0:3] for img in images]
    imageio.mimsave('images/first_pointcloud_360.gif', img_list, loop=10, duration = 0.05)
    return img_list[0], points, rgb


def render_second_pointcloud(
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color)
    rgb_data = load_rgbd_data()

    points, rgb = unproject_depth_image(rgb_data.get("rgb2"), rgb_data.get("mask2"),
                                        rgb_data.get("depth2"), rgb_data.get("cameras2"))
    # extend points to 12 views
    num_views = 12
    points = points.unsqueeze(0).to(device)
    rgb = rgb.unsqueeze(0).to(device)
    points = points.expand(num_views, -1, -1)
    rgb = rgb.expand(num_views, -1, -1)
    point_cloud = pytorch3d.structures.Pointclouds(points=points, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=6,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )

    # Additional rotation of 180 degrees about the z-axis
    additional_rotation = torch.tensor([[-1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, 1]], dtype=R.dtype, device=R.device)
    additional_rotation = additional_rotation.unsqueeze(0).expand(num_views, -1, -1)

    # Apply additional rotation to the original rotations
    R_additional = R @ additional_rotation

    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_additional,
        T=T,
        device=device
    )

    images = renderer(point_cloud, cameras=many_cameras)
    images = images.cpu().numpy()
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255)[:,:,0:3] for img in images]
    imageio.mimsave('images/second_pointcloud_360.gif', img_list, loop=10, duration = 0.05)
    return img_list[0], points, rgb


def render_composite_cloud(
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color)
    _, pts1, features1 = render_first_pointcloud(image_size=image_size, device=device)
    _, pts2, features2 = render_second_pointcloud(image_size=image_size, device=device)

    import ipdb
    ipdb.set_trace()
    num_views = 12

    combined_points = torch.cat([pts1, pts2], dim=1)
    combined_features = torch.cat([features1, features2], dim=1)
    combined_point_cloud = pytorch3d.structures.Pointclouds(points=combined_points, features=combined_features)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=6,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )

    # Additional rotation of 180 degrees about the z-axis
    additional_rotation = torch.tensor([[-1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, 1]], dtype=R.dtype, device=R.device)
    additional_rotation = additional_rotation.unsqueeze(0).expand(num_views, -1, -1)

    # Apply additional rotation to the original rotations
    R_additional = R @ additional_rotation

    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_additional,
        T=T,
        device=device
    )

    images = renderer(combined_point_cloud, cameras=many_cameras)
    images = images.cpu().numpy()
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255)[:,:,0:3] for img in images]
    imageio.mimsave('images/composite_pointcloud_360.gif', img_list, loop=10, duration = 0.05)
    return img_list[0]


def render_bridge(
    point_cloud_path="data/bridge_pointcloud.npz",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend


def render_sphere(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = torch.sin(Theta) * torch.cos(Phi)
    y = torch.cos(Theta)
    z = torch.sin(Theta) * torch.sin(Phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()


def render_torus(image_size=256, num_samples=200, device=None):
    """
    Renders a Torus
    """
    start = time.time()
    background_color=(1, 1, 1)
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
    image_size=image_size, background_color=background_color)
    R = 1
    r = 0.5
    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    Phi, Theta = torch.meshgrid(phi, theta)
    x = (R + r * torch.cos(Phi)) * torch.cos(Theta)
    y = (R + r * torch.cos(Phi)) * torch.sin(Theta)
    z = r * torch.sin(Phi)

    num_views = 12
    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    points = points.unsqueeze(0).expand(num_views, -1, -1)
    color = color.unsqueeze(0).expand(num_views, -1, -1)

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=points, features=color,)
    print("Memory occupied by torus_point_cloud in MB: ", sys.getsizeof(torus_point_cloud) / (1024 * 1024))
    torus_point_cloud = torus_point_cloud.to(device)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=6,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )

    # Additional rotation of 20 degrees about the x-axis
    additional_rotation = torch.tensor([[1,  0, 0],
                                        [0, np.cos(np.radians(20)), np.sin(np.radians(20))],
                                        [0,  -np.sin(np.radians(20)), np.cos(np.radians(20))]],
                                        dtype=R.dtype, device=R.device)
    additional_rotation = additional_rotation.unsqueeze(0).expand(num_views, -1, -1)

    # Apply additional rotation to the original rotations
    R_additional = R @ additional_rotation

    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_additional,
        T=T,
        device=device
    )

    images = renderer(torus_point_cloud, cameras=many_cameras)
    end = time.time()
    print("Time taken to render parametric torus: ", end - start)
    images = images.cpu().numpy()
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255)[:,:,0:3] for img in images]
    imageio.mimsave('images/torus_pointcloud_360.gif', img_list, loop=10, duration = 0.05)
    return img_list[0]

    """
    Memory occupied by torus_point_cloud in MB:  4.57763671875e-05
    Time taken to render parametric torus:  0.31734442710876465
    """


def render_custom_shape(image_size=256, num_samples=200, device=None):
    """
    Renders a new shape
    """
    background_color=(1, 1, 1)
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
    image_size=image_size, background_color=background_color)

    def hyperboloid(u, v, a=1):
        x = a * np.cosh(u) * np.cos(v)
        y = a * np.cosh(u) * np.sin(v)
        z = a * np.sinh(u)
        return x, y, z

    # Generate u and v values
    u = torch.linspace(-2, 2, 100)
    v = torch.linspace(0, 2 * np.pi, 100)
    u, v = torch.meshgrid(u, v)

    # Generate hyperboloid points
    x, y, z = hyperboloid(u, v)

    num_views = 12
    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    points = points.unsqueeze(0).expand(num_views, -1, -1)
    color = color.unsqueeze(0).expand(num_views, -1, -1)

    hyperboloid_point_cloud = pytorch3d.structures.Pointclouds(
        points=points, features=color,
    ).to(device)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=6,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )

    # Additional rotation of 10 degrees about the x-axis
    additional_rotation = torch.tensor([[1,  0, 0],
                                        [0, np.cos(np.radians(80)), np.sin(np.radians(80))],
                                        [0,  -np.sin(np.radians(80)), np.cos(np.radians(80))]],
                                        dtype=R.dtype, device=R.device)
    additional_rotation = additional_rotation.unsqueeze(0).expand(num_views, -1, -1)

    # Apply additional rotation to the original rotations
    R_additional = R @ additional_rotation

    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_additional,
        T=T,
        device=device
    )

    images = renderer(hyperboloid_point_cloud, cameras=many_cameras)
    images = images.cpu().numpy()
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255)[:,:,0:3] for img in images]
    imageio.mimsave('images/custom_hyperbolloid_pointcloud_360.gif', img_list, loop=10, duration = 0.05)
    return img_list[0]


def render_sphere_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)


def render_torus_implicit(image_size=256, voxel_size=64, device=None):
    start = time.time()
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    R = 0.6
    r = 0.4
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = (torch.sqrt(X ** 2 + Y ** 2) - R) ** 2 + Z ** 2 - r ** 2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())

    num_views = 12
    vertices = vertices.unsqueeze(0).expand(num_views, -1, -1)
    faces = faces.unsqueeze(0).expand(num_views, -1, -1)
    textures = textures.unsqueeze(0).expand(num_views, -1, -1)

    textures = pytorch3d.renderer.TexturesVertex(vertices)

    mesh = pytorch3d.structures.Meshes(vertices, faces, textures=textures)
    print("Memory occupied by mesh in MB: ", sys.getsizeof(mesh) / (1024 * 1024))
    mesh = mesh.to(device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    # R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=6,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )

    # Additional rotation of 10 degrees about the x-axis
    additional_rotation = torch.tensor([[1,  0, 0],
                                        [0, np.cos(np.radians(20)), np.sin(np.radians(20))],
                                        [0,  -np.sin(np.radians(20)), np.cos(np.radians(20))]],
                                        dtype=R.dtype, device=R.device)
    additional_rotation = additional_rotation.unsqueeze(0).expand(num_views, -1, -1)

    # Apply additional rotation to the original rotations
    R_additional = R @ additional_rotation

    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_additional,
        T=T,
        device=device
    )

    images = renderer(mesh, cameras=many_cameras, lights=lights)
    end = time.time()
    print("Time taken to render implicit torus: ", end - start)
    images = images.cpu().numpy()
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255)[:,:,0:3] for img in images]
    imageio.mimsave('images/implicit_torus_360.gif', img_list, loop=10, duration = 0.05)
    return img_list[0]

    """
    Memory occupied by mesh in MB:  4.57763671875e-05
    Time taken to render implicit torus:  2.4901247024536133
    """


def render_custom_implicit(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    a = 0.6
    b = 0.6
    c = 0.8
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 / a ** 2 + Y ** 2 / b ** 2 - Z ** 2 / c ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())

    num_views = 12
    vertices = vertices.unsqueeze(0).expand(num_views, -1, -1)
    faces = faces.unsqueeze(0).expand(num_views, -1, -1)
    textures = textures.unsqueeze(0).expand(num_views, -1, -1)

    textures = pytorch3d.renderer.TexturesVertex(vertices)

    mesh = pytorch3d.structures.Meshes(vertices, faces, textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    # R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=6,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )

    # Additional rotation of 10 degrees about the x-axis
    additional_rotation = torch.tensor([[1,  0, 0],
                                        [0, np.cos(np.radians(20)), np.sin(np.radians(20))],
                                        [0,  -np.sin(np.radians(20)), np.cos(np.radians(20))]],
                                        dtype=R.dtype, device=R.device)
    additional_rotation = additional_rotation.unsqueeze(0).expand(num_views, -1, -1)

    # Apply additional rotation to the original rotations
    R_additional = R @ additional_rotation

    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_additional,
        T=T,
        device=device
    )

    images = renderer(mesh, cameras=many_cameras, lights=lights)
    images = images.cpu().numpy()
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255)[:,:,0:3] for img in images]
    imageio.mimsave('images/implicit_hyperbolloid_360.gif', img_list, loop=10, duration = 0.05)
    return img_list[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "point_cloud_first", "point_cloud_second",
                 "point_cloud_composite", "torus", "custom_shape", "parametric",
                 "implicit", "torus_implicit", "custom_implicit"],
    )
    parser.add_argument("--output_path", type=str, default="images/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    if args.render == "point_cloud":
        image = render_bridge(image_size=args.image_size)
    elif args.render == "point_cloud_first":
        image, _, _ = render_first_pointcloud(image_size=args.image_size)
    elif args.render == "point_cloud_second":
        image, _, _ = render_second_pointcloud(image_size=args.image_size)
    elif args.render == "point_cloud_composite":
        image = render_composite_cloud(image_size=args.image_size)
    elif args.render == "parametric":
        image = render_sphere(image_size=args.image_size, num_samples=args.num_samples)
    elif args.render == "torus":
        image = render_torus(image_size=args.image_size, num_samples=args.num_samples)
    elif args.render == "custom_shape":
        image = render_custom_shape(image_size=args.image_size, num_samples=args.num_samples)
    elif args.render == "implicit":
        image = render_sphere_mesh(image_size=args.image_size)
    elif args.render == "torus_implicit":
        image = render_torus_implicit(image_size=args.image_size)
    elif args.render == "custom_implicit":
        image = render_custom_implicit(image_size=args.image_size)
    else:
        raise Exception("Did not understand {}".format(args.render))
    plt.imsave(args.output_path, image)

