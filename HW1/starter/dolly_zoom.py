"""
Usage:
    python -m starter.dolly_zoom --num_frames 10
"""

import argparse

import imageio
import numpy as np
import pytorch3d
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from starter.utils import get_device, get_mesh_renderer


def dolly_zoom(
    image_size=256,
    num_frames=10,
    duration=3,
    device=None,
    output_file="output/dolly.gif",
):
    if device is None:
        device = get_device()

    mesh = pytorch3d.io.load_objs_as_meshes(["data/cow_on_plane.obj"])
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    fovs = torch.linspace(5, 120, num_frames)

    """
    - FOV and focal length are inversely proportional.
    - As the FOV increases, the focal length decreases.
    - Hence, to maintain the object size in the image, we need to move the camera closer to the object.
    - Formula is distance = distance * (tan(prev_fov/2) / tan(fov/2))
    """
    distance = 50.0

    renders = []
    for i in tqdm(range(1,num_frames)):
        fov = fovs[i].cpu().item()
        prev_fov = fovs[i-1].cpu().item()
        # Calculate the new distance to maintain object size constant
        distance = distance * (np.tan(np.radians(prev_fov/2)) / np.tan(np.radians(fov/2)))
        T = [[0, 0, distance]]
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov, T=T, device=device)
        rend = renderer(mesh.extend(num_frames), cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"fov: {fovs[i]:.2f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, fps=(num_frames / duration), loop=3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=30)
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--output_file", type=str, default="images/dolly.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    dolly_zoom(
        image_size=args.image_size,
        num_frames=args.num_frames,
        duration=args.duration,
        output_file=args.output_file,
    )
