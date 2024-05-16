import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle, # NOTE ray_bundle is a class defined in ray_utils.py
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray, device=ray_bundle.origins.device)
        # z_vals.shape = (self.n_pts_per_ray,)

        # TODO (Q1.4): Sample points from z values
        """
        NOTE: if image_plane_points.shape = torch.Size([65536, 3]),
              then rays_origin.shape = torch.Size([65536, 3])
              and sample_lenths.shape = torch.Size([65536, 1, 3])
        """

        origins_expanded = ray_bundle.origins.unsqueeze(1)  # Shape: (N, 1, 3)
        origins_expanded = origins_expanded.expand(-1, self.n_pts_per_ray, -1)  # Shape: (N, D, 3)
        directions_expanded = ray_bundle.directions.unsqueeze(1)  # Shape: (N, 1, 3)
        directions_expanded = directions_expanded.expand(-1, self.n_pts_per_ray, -1)  # Shape: (N, D, 3)
        # convert z_vals to shape Shape: (1, D, 1)
        z_vals_expanded = z_vals.expand(ray_bundle.origins.shape[0], -1).unsqueeze(-1)  # Shape: (1, D, 1)

        # Compute sample points
        # (N, D, 3) = (N, 1, 3) + (1, D, 1) * (N, 1, 3)
        new_sample_points = origins_expanded + z_vals_expanded * directions_expanded

        # Return
        return ray_bundle._replace(
            sample_points=new_sample_points,
            sample_lengths=z_vals_expanded * torch.ones_like(new_sample_points[..., :1]), # shape = (N, D, 1)
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}