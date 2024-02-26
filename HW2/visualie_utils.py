import torch
from pytorch3d.ops import cubify
from render_utils import *
from pytorch3d.utils import ico_sphere
import pytorch3d
import numpy as np
import imageio
import matplotlib.pyplot as plt
import mcubes
import utils_vox
from pytorch3d.ops import sample_points_from_meshes

def visualize_voxels_as_mesh(voxel_grid, image_name):
    device = get_device()

    vertices, faces = mcubes.marching_cubes(voxel_grid.detach().cpu().squeeze().numpy(), isovalue=0.5)
    n_verts = vertices.shape[0]
    vertices = torch.tensor(vertices).float().unsqueeze(0)
    faces = torch.tensor(faces.astype(int)).unsqueeze(0)
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures_tensor = torch.ones_like(vertices).to(device)
    textures = pytorch3d.renderer.TexturesVertex(textures_tensor).to(device)

    mesh = pytorch3d.structures.Meshes(vertices, faces, textures=textures).to(device)
    mesh_offset = -1*vertices.mean(dim=1).to(device)
    mesh = mesh.offset_verts(mesh_offset.repeat(n_verts,1)) # zero center the mesh
    renderer = get_mesh_renderer(image_size=1024)
    mesh = mesh.to(device)

    num_views = 12
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=2,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    images = renderer(mesh.extend(num_views), cameras=many_cameras, lights=lights)
    fig, axs = plt.subplots(3, 4)
    axs = axs.flatten()
    for i, image in enumerate(images):
        ax = axs[i]
        ax.imshow(image.cpu())
        ax.axis("off")

    plt.savefig(f'images/voxels_{image_name}.png')
    # Optionally, show the figure
    # plt.show()
    images = images.cpu().numpy()
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255) for img in images]
    imageio.mimsave(f'images/voxels_{image_name}.gif', img_list, loop=10, duration = 0.05)
    plt.close()

def visualize_voxels(voxel_grid, image_name):
    device = get_device()
    # Assume voxel_grid is your voxel tensor of shape (h x w x d)
    # voxel_grid = ...

    # convert to (1 x h x w x d) tensor
    # voxel_grid = torch.unsqueeze(voxel_grid, 0)

    # Convert the voxel grid to a mesh
    threshold = 0.5  # adjust this value as needed
    mesh = cubify(voxel_grid, threshold)

    verts_packed = mesh.verts_packed()
    color=[0.7, 0.7, 1]
    textures = torch.ones_like(verts_packed.unsqueeze(0))
    textures = textures * torch.tensor(color).to(device)
    mesh.textures = pytorch3d.renderer.TexturesVertex(textures)

    renderer = get_mesh_renderer(image_size=512)
    mesh = mesh.to(device)

    num_views = 12
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=2,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    images = renderer(mesh.extend(num_views), cameras=many_cameras, lights=lights)
    fig, axs = plt.subplots(3, 4)
    axs = axs.flatten()
    for i, image in enumerate(images):
        ax = axs[i]
        ax.imshow(image.cpu())
        ax.axis("off")

    plt.savefig(f'images/voxels_{image_name}.png')
    # Optionally, show the figure
    # plt.show()
    images = images.cpu().numpy()
    # img_list = [img.squeeze().astype(np.uint8) for img in images]
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255) for img in images]
    imageio.mimsave(f'images/voxels_{image_name}.gif', img_list, loop=10, duration = 0.05)
    plt.close()


def visualize_point_difference(pred, mesh_gt, image_name):
    device = get_device()
    pred_points = pred[0]
    gt_points = sample_points_from_meshes(mesh_gt, 3000)[0]

    pred_points = pred_points.to(device)
    gt_points = gt_points.to(device)

    # find points in pred_points which are outliers to gt_points and store them as a separate pointcloud
    pred_to_gt_dists = torch.cdist(pred_points, gt_points, p=2)
    pred_to_gt_dists, _ = pred_to_gt_dists.min(dim=1)
    pred_to_gt_dists = pred_to_gt_dists / pred_to_gt_dists.max()
    pred_outliers = pred_points[pred_to_gt_dists > 0.1]

    renderer = get_points_renderer(
    image_size=1024, background_color=(1,1,1), device=device)
    num_views = 12

    # create a color tensor for pred_outliers which is red and for gt_points which is blue
    color_pred_outliers = torch.tensor([1, 0, 0], dtype=torch.float32).expand(pred_outliers.shape[0], -1)
    color_gt_points = torch.tensor([0, 0, 1], dtype=torch.float32).expand(gt_points.shape[0], -1)

    # combine the two pointclouds and their colors
    points = torch.cat([pred_outliers, gt_points], dim=0).to(device)
    color = torch.cat([color_pred_outliers, color_gt_points], dim=0).to(device)

    # convert to (num_views x n_points x 3) tensor
    points = points.unsqueeze(0).expand(num_views, -1, -1)
    color = color.unsqueeze(0).expand(num_views, -1, -1)

    render_point_cloud = pytorch3d.structures.Pointclouds(
        points=points, features=color,)
    render_point_cloud = render_point_cloud.to(device)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=1.5,
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

    images = renderer(render_point_cloud, cameras=many_cameras)
    images = images.cpu().numpy()
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255)[:,:,0:3] for img in images]
    imageio.mimsave(f'images/pointcloud_{image_name}.gif', img_list, loop=10, duration = 0.05)
    return img_list[0]

def visualize_pointcloud(point_cloud, image_name):
    """
    point_cloud: n_points x 3
    """
    device = get_device()
    renderer = get_points_renderer(
    image_size=1024, background_color=(1,1,1), device=device)

    num_views = 12
    points = point_cloud[0]
    color = (points - points.min()) / (points.max() - points.min())

    # convert to (num_views x n_points x 3) tensor
    points = points.unsqueeze(0).expand(num_views, -1, -1)
    color = color.unsqueeze(0).expand(num_views, -1, -1)

    render_point_cloud = pytorch3d.structures.Pointclouds(
        points=points, features=color,)
    render_point_cloud = render_point_cloud.to(device)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=1.5,
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

    images = renderer(render_point_cloud, cameras=many_cameras)
    images = images.cpu().numpy()
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255)[:,:,0:3] for img in images]
    imageio.mimsave(f'images/pointcloud_{image_name}.gif', img_list, loop=10, duration = 0.05)
    return img_list[0]

def visualize_mesh(input_mesh, image_name):
    """
    input_mesh: Meshes
    """
    device = get_device()
    renderer = get_mesh_renderer(image_size=1024)

    num_views = 12
    mesh = input_mesh

    verts_packed = mesh.verts_packed()
    color=[0.7, 0.7, 1]
    textures = torch.ones_like(verts_packed.unsqueeze(0)).to(device)
    textures = textures * torch.tensor(color).to(device)
    mesh.textures = pytorch3d.renderer.TexturesVertex(textures)

    mesh = mesh.to(device)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=1.5,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    images = renderer(mesh.extend(num_views), cameras=many_cameras, lights=lights)
    fig, axs = plt.subplots(3, 4)
    axs = axs.flatten()
    for i, image in enumerate(images):
        ax = axs[i]
        ax.imshow(image.cpu())
        ax.axis("off")

    plt.savefig(f'images/mesh_{image_name}.png')
    # Optionally, show the figure
    # plt.show()
    images = images.cpu().numpy()
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255) for img in images]
    imageio.mimsave(f'images/mesh_{image_name}.gif', img_list, loop=10, duration = 0.05)
    plt.close()


def render_mesh_to_img(input_mesh, image_name):
    """
    input_mesh: Meshes
    """
    device = get_device()
    renderer = get_mesh_renderer(image_size=1024)

    num_views = 12
    mesh = input_mesh

    verts_packed = mesh.verts_packed()
    color=[0.7, 0.7, 1]
    textures = torch.ones_like(verts_packed.unsqueeze(0)).to(device)
    textures = textures * torch.tensor(color).to(device)
    mesh.textures = pytorch3d.renderer.TexturesVertex(textures)

    mesh = mesh.to(device)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=1.5,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    images = renderer(mesh.extend(num_views), cameras=many_cameras, lights=lights)

    # Select the 7th image (index 6)
    image = images[6]

    # Display and save the image
    plt.imshow(image.cpu())
    plt.axis("off")
    plt.savefig(f'images/mesh_{image_name}.png')
    plt.close()