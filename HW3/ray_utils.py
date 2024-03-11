import math
from typing import List, NamedTuple

import torch
from pytorch3d.renderer.cameras import CamerasBase


# Convenience class wrapping several ray inputs:
#   1) Origins -- ray origins
#   2) Directions -- ray directions
#   3) Sample points -- sample points along ray direction from ray origin
#   4) Sample lengths -- distance of sample points from ray origin

class RayBundle(object):
    def __init__(
        self,
        origins,
        directions,
        sample_points,
        sample_lengths,
    ):
        self.origins = origins
        self.directions = directions
        self.sample_points = sample_points
        self.sample_lengths = sample_lengths

    def __getitem__(self, idx):
        return RayBundle(
            self.origins[idx],
            self.directions[idx],
            self.sample_points[idx],
            self.sample_lengths[idx],
        )

    @property
    def shape(self):
        return self.origins.shape[:-1]

    @property
    def sample_shape(self):
        return self.sample_points.shape[:-1]

    def reshape(self, *args):
        return RayBundle(
            self.origins.reshape(*args, 3),
            self.directions.reshape(*args, 3),
            self.sample_points.reshape(*args, self.sample_points.shape[-2], 3),
            self.sample_lengths.reshape(*args, self.sample_lengths.shape[-2], 3),
        )

    def view(self, *args):
        return RayBundle(
            self.origins.view(*args, 3),
            self.directions.view(*args, 3),
            self.sample_points.view(*args, self.sample_points.shape[-2], 3),
            self.sample_lengths.view(*args, self.sample_lengths.shape[-2], 3),
        )

    def _replace(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        return self


# Sample image colors from pixel values
def sample_images_at_xy(
    images: torch.Tensor,
    xy_grid: torch.Tensor,
):
    batch_size = images.shape[0]
    spatial_size = images.shape[1:-1]

    xy_grid = -xy_grid.view(batch_size, -1, 1, 2).to(device=images.device)

    images_sampled = torch.nn.functional.grid_sample(
        images.permute(0, 3, 1, 2),
        xy_grid,
        align_corners=True,
        mode="bilinear",
    )

    return images_sampled.permute(0, 2, 3, 1).view(-1, images.shape[-1])


# Generate pixel coordinates from in NDC space (from [-1, 1])
def get_pixels_from_image(image_size, camera):
    W, H = image_size[0], image_size[1]

    # TODO (Q1.3): Generate pixel coordinates from [0, W] in x and [0, H] in y
    # TODO (Q1.3): Convert to the range [-1, 1] in both x and y
    x = torch.linspace(-1, 1, steps=W, dtype=torch.float32)
    y = torch.linspace(-1, 1, steps=H, dtype=torch.float32)

    # Create grid of coordinates
    xy_grid = torch.stack(
        tuple( reversed( torch.meshgrid(y, x) ) ),
        dim=-1,
    ).view(W * H, 2)

    return -xy_grid


# Random subsampling of pixels from an image
def get_random_pixels_from_image(n_pixels, image_size, camera):
    xy_grid = get_pixels_from_image(image_size, camera)
    # NOTE: xy_grid is of shape (W * H, 2)

    # TODO (Q2.1): Random subsampling of pixel coordinaters
    # Create random mask of xy_grid
    mask = torch.randperm(xy_grid.shape[0])
    xy_grid_sub = xy_grid[mask]

    return xy_grid_sub.reshape(-1, 2)[:n_pixels] # shape = (num grid_pts, x and y)


# Get rays from pixel values
def get_rays_from_pixels(xy_grid, image_size, camera):
    W, H = image_size[0], image_size[1]

    """
    Projection of [XYZ] to [xy] is given by the camera parameters (intrinsic and extrinsic) as:
        # for perspective camera (What we'll use)
        x = fx * X / Z + px
        y = fy * Y / Z + py
        z = 1 / Z

        # But, we have an additional stage where we can resacle objects such that all objects are in the range [-1, 1]
        # This is called Normalized Device Coordinates (NDC) space

        In this case, Z = 1 and we can use camera.unproject_points to go from NDC space the world space points
        NOTE: camera object is of type pytorch3d.CamerasBase
    """
    # TODO (Q1.3): Map pixels to points on the image plane at Z=1
    ndc_points = xy_grid.to(device=camera.device)

    ndc_points = torch.cat(
        [
            ndc_points,
            torch.ones_like(ndc_points[..., -1:], device=camera.device)
        ],
        dim=-1
    )

    # TODO (Q1.3): Use camera.unproject to get world space points from NDC space points
    image_plane_points = camera.unproject_points(ndc_points, from_ndc=True)
    # ipdb> image_plane_points[1]
    # tensor([ 0.9922,  1.0000, -2.0000], device='cuda:0')

    # TODO (Q1.3): Get ray origins from camera center
    # origin is of shape (1,3) i.e. just a point in 3D, we'll expand that to be the ray origin for all
    # points on the image plane
    rays_origin = camera.get_camera_center().expand(image_plane_points.shape[0], -1)
    # assert rays_origin.shape == image_plane_points.shape -> THIS SHOULD BE TRUE

    # TODO (Q1.3): Get ray directions as image_plane_points - rays_origin
    rays_d = torch.nn.functional.normalize(image_plane_points - rays_origin)

    # Create and return RayBundle
    """
    NOTE: if image_plane_points.shape = torch.Size([65536, 3]),
          then rays_origin.shape = torch.Size([65536, 3])
          and sample_lenths.shape = torch.Size([65536, 1, 3])
    """
    return RayBundle(
        rays_origin,
        rays_d,
        sample_lengths=torch.zeros_like(rays_origin).unsqueeze(1),
        sample_points=torch.zeros_like(rays_origin).unsqueeze(1),
    )