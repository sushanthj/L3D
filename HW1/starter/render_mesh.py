"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pytorch3d
import torch
import imageio

from torchvision import transforms

from PIL import Image
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesUV,
    TexturesVertex,
)

def render_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend

def render_cow_multiple_views(
    cow_path="data/cow.obj",
    image_size=256,
    color=[0.7, 0.7, 1],
    device=None):
        if device is None:
            device = get_device()
        # Get the renderer.
        renderer = get_mesh_renderer(image_size=image_size)

        # Get the vertices, faces, and textures.
        vertices, faces = load_cow_mesh(cow_path)
        vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
        faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
        textures = torch.ones_like(vertices)  # (1, N_v, 3)
        textures = textures * torch.tensor(color)  # (1, N_v, 3)
        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
        )
        mesh = mesh.to(device)

        num_views = 12
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=3,
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

        plt.savefig('images/trial_image.png')
        # Optionally, show the figure
        # plt.show()
        images = images.cpu().numpy()
        # img_list = [img.squeeze().astype(np.uint8) for img in images]
        img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255) for img in images]
        imageio.mimsave('images/cow_360.gif', img_list, loop=10, duration = 0.05)
        plt.close()

def render_tetrahedron(image_size=256, color=[0.7, 0.7, 1], device=None):
    if device is None:
            device = get_device()
    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    vertices = torch.tensor([[3,0,0],[0,0,3],[-2,0,-2],[0,3,0]], dtype=torch.float32).unsqueeze(0)
    faces = torch.tensor([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=torch.int64).unsqueeze(0)

    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    num_views = 12
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=6,
        elev=3,
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

    # Save the figure to a file (adjust the file format as needed)
    plt.savefig('images/tetrahedron.png')
    # Optionally, show the figure
    # plt.show()
    images = images.cpu().numpy()
    # img_list = [img.squeeze().astype(np.uint8) for img in images]
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255) for img in images]

    imageio.mimsave('images/tetrahedron_360.gif', img_list, loop=10, duration = 0.05)
    plt.close()

def render_cube(image_size=256, color=[0.7, 0.7, 1], device=None):
    if device is None:
            device = get_device()
    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # define the vertices for a cube
    cube_coords = [
        [3, 3, 3],
        [3, 3, -3],
        [3, -3, 3],
        [3, -3, -3],
        [-3, 3, 3],
        [-3, 3, -3],
        [-3, -3, 3],
        [-3, -3, -3],
    ]
    cube_faces = [
        [0, 1, 2],
        [1, 2, 3],
        [4, 5, 6],
        [5, 6, 7],
        [0, 1, 4],
        [1, 4, 5],
        [2, 3, 6],
        [3, 6, 7],
        [0, 2, 4],
        [2, 4, 6],
        [1, 3, 5],
        [3, 5, 7],
    ]
    vertices = torch.tensor(cube_coords, dtype=torch.float32).unsqueeze(0)
    faces = torch.tensor(cube_faces, dtype=torch.int64).unsqueeze(0)

    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    num_views = 12
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=15,
        elev=3,
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

    # Save the figure to a file (adjust the file format as needed)
    plt.savefig('images/cube.png')
    # Optionally, show the figure
    # plt.show()
    images = images.cpu().numpy()
    # img_list = [img.squeeze().astype(np.uint8) for img in images]
    img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255) for img in images]

    imageio.mimsave('images/cube_360.gif', img_list, loop=10, duration = 0.05)
    plt.close()

def render_bb8(
    cow_path="data/bb8.obj",
    image_size=256,
    color=[0.7, 0.7, 1],
    device=None):
        if device is None:
            device = get_device()
        # Get the renderer.
        renderer = get_mesh_renderer(image_size=image_size, bin_size=0)

        # Get the vertices, faces, and textures.
        vertices, faces = load_cow_mesh(cow_path)
        vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
        faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
        textures = torch.ones_like(vertices)  # (1, N_v, 3)
        textures = textures * torch.tensor(color)  # (1, N_v, 3)
        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
        )
        mesh = mesh.to(device)

        num_views = 12
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=10,
            elev=50,
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

        plt.savefig('images/trial_image.png')
        # Optionally, show the figure
        # plt.show()
        images = images.cpu().numpy()
        # img_list = [img.squeeze().astype(np.uint8) for img in images]
        img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255) for img in images]
        imageio.mimsave('images/bb8_360.gif', img_list, loop=10, duration = 0.05)
        plt.close()


def retexture_cow(
    cow_path="data/cow.obj",
    image_size=256,
    color1=[0.0, 0.4, 0.6],
    color2=[0.6, 0.4, 0.0],
    device=None):
        """
        color1=[0.0, 0.4, 0.6],
        color2=[0.6, 0.4, 0.0],

        Both are blended as per the z value of the vertices
        """
        if device is None:
            device = get_device()
        # Get the renderer.
        renderer = get_mesh_renderer(image_size=image_size)

        # Get the vertices, faces, and textures.
        vertices, faces = load_cow_mesh(cow_path)
        vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
        faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

        ######### Blend color according to z value of the vertices #######
        textures = vertices[0].clone()
        z_min = vertices[..., 2].min().cpu().item()
        z_max = vertices[..., 2].max().cpu().item()
        print("Z min:", z_min, "Z max:", z_max)
        color1 = torch.tensor(color1, dtype=torch.float32).to(device)
        color2 = torch.tensor(color2, dtype=torch.float32).to(device)
        alpha = lambda z: (z - z_min) / (z_max - z_min)
        color = lambda alpha: alpha * color2 + (1 - alpha) * color1

        for i in range(textures.shape[0]): # textures shape = (N_v, 3)
            textures[i] = color(alpha(textures[i, 2].cpu().item()))
        textures = textures.unsqueeze(0)
        ##################################################################

        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
        )
        mesh = mesh.to(device)

        num_views = 12
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=3,
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

        plt.savefig('images/trial_image.png')
        # Optionally, show the figure
        # plt.show()
        images = images.cpu().numpy()
        # img_list = [img.squeeze().astype(np.uint8) for img in images]
        img_list = [np.clip((img.squeeze() * 255).astype(np.uint8), 0, 255) for img in images]
        imageio.mimsave('images/cow_retextured_360.gif', img_list, loop=10, duration = 0.05)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/cow_render.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    # image = render_cow(cow_path=args.cow_path, image_size=args.image_size)
    # plt.imsave(args.output_path, image)

    # image = render_cow_multiple_views(cow_path=args.cow_path, image_size=args.image_size)
    # render_tetrahedron(image_size=args.image_size)
    # retexture_cow(cow_path=args.cow_path, image_size=args.image_size)
    # render_bb8(image_size=args.image_size)
    render_cube(image_size=args.image_size)
