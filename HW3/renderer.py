import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10):
        """

        Args:
            deltas : distance between each sample (self._chunk_size, n_pts, 1)
            rays_density (torch.Tensor): (self._chunk_size, n_pts, 1) predicting density values of each sample (from NERF MLP)
            eps (float, optional): Defaults to 1e-10.

        Returns:
            _type_: _description_
        """
        # TODO (1.5): Compute transmittance using the equation described in the README
        num_rays, num_sample_points, _ = deltas.shape
        transmittances = []
        transmittances.append(torch.ones((num_rays, 1)).to(deltas.device)) # first transmittance is 1

        #! Find the transmittance for each discrete volume
        for i in range(1, num_sample_points):
            # recursive formula for transmittance
            transmittances.append(transmittances[i-1] * torch.exp(-rays_density[:, i-1] * deltas[:, i-1] + eps))

        # TODO (1.5): Compute weight used for rendering from transmittance and alpha
        #! Multiply transmittance with the (1-e^(-sigma(x)*delta_x) part of the equation
        transmittances_stacked = torch.stack(transmittances, dim=1)
        # the below line implements the T(x, x_t) * (1 - e^{−σ(x) * δx}) part of the equation => we'll call this 'weights'
        return transmittances_stacked * (1 - torch.exp(-rays_density*deltas+eps)) # -> weights

    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor):
        """

        Args:
            weights (torch.Tensor): (self._chunk_size, n_pts, 1) (Overall Transmittance for each ray)
            rays_feature (torch.Tensor): (self._chunk_size*n_pts, 3) rays_feature = RGB color or depth

        Returns:
            feature : Final Attribute (color or depth) for each ray
        """
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        feature = torch.sum((weights*rays_feature), dim=1)
        return feature


    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle):
        """

        Args:
            sampler : Samples points along rays ( see StratifiedRaysampler in sampler.py)
            implicit_fn : Implicit function ( see SphereTracingSDF in implicit.py)
            ray_bundle : Ray Bundle object with ray origins and directions of shape  (N, 3) where N = number of rays
                         and sample points and lengths of shape (N, n_pts_per_ray, 3)

        Returns:
            out : Returns color and depth information for each ray
        """
        B = ray_bundle.shape[0] # ray_bundle.shape = (N,)

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1] # number of sample pts along each ray

            # Call implicit function with sample points (NERF MLP is the implicit function here)
            implicit_output = implicit_fn(cur_ray_bundle) # gives the signed_distance for the query points
            predicted_density_for_all_samples_for_all_rays_in_chunk = implicit_output['density'] # shape = (self._chunk_size*n_pts, 1) : The density value of that discrete volume
            predicted_colors_for_all_samples_for_all_rays_in_chunk = implicit_output['feature'] # shape = (self._chunk_size*n_pts, 3) : Emittance for each discrete volume for RGB channels

            # Compute length of each ray segment
            # NOTE: cur_ray_bundle.sample_lengths.shape = (self._chunk_size, n_pts, n_pts)
            depth_values = cur_ray_bundle.sample_lengths[..., 0] # depth_values.shape = (self._chunk_size, n_pts)
            # deltas are the distance between each sample
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights (weights = overall transmittance for all rays in the chunk)
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1), # shape = (self._chunk_size, n_pts, 1)
                predicted_density_for_all_samples_for_all_rays_in_chunk.view(-1, n_pts, 1) # shape = (self._chunk_size, n_pts, 1)
            )

            # TODO (1.5): Render (color) features using weights
            # weights.shape = (self._chunk_size, n_pts, 1)
            # color.shape = (self._chunk_size*n_pts, 3)
            color_of_all_rays = self._aggregate(weights, predicted_colors_for_all_samples_for_all_rays_in_chunk.view(-1, n_pts, 3)) # feature = RGB color

            # TODO (1.5): Render depth map
            # depth_values.shape = (self._chunk_size, n_pts)
            depth_of_all_rays = self._aggregate(weights, depth_values.view(-1, n_pts, 1))

            # Return
            cur_out = {
                'feature': color_of_all_rays,
                'depth': depth_of_all_rays,
            }
            # shape = (N, 3) for feature and (N, 1) for depth

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class SphereTracingRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self.near = cfg.near
        self.far = cfg.far
        self.max_iters = cfg.max_iters

    def sphere_tracing(
        self,
        implicit_fn,
        origins, # Nx3
        directions, # Nx3
    ):
        '''
        Input:
            implicit_fn: a module that computes a SDF at a query point
            origins: N_rays X 3
            directions: N_rays X 3
        Output:
            points: N_rays X 3 points indicating ray-surface intersections. For rays that do not intersect the surface,
                    the point can be arbitrary.
            mask: N_rays X 1 (boolean tensor) denoting which of the input rays intersect the surface.
        '''
        # TODO (Q5): Implement sphere tracing
        # 1) Iteratively update points and distance to the closest surface
        #   in order to compute intersection points of rays with the implicit surface
        # 2) Maintain a mask with the same batch dimension as the ray origins,
        #   indicating which points hit the surface, and which do not

        # Use formula : points = origins + t*directions
        # init t vector (N, 1) and points (N, 3)
        t = torch.zeros(origins.shape[0], 1).to(origins.device)
        points = torch.zeros_like(origins)
        # define a threshold to stop the sphere tracing
        threshold = 1e-8

        #? Use self.near and self.far too?

        for _ in range(self.max_iters):
            points = origins + t*directions

            distance_to_surface = implicit_fn(points).view(origins.shape[0], 1)
            mask = torch.where(distance_to_surface < threshold, True, False)

            # if all rays hit the surface, break out of loop
            if mask.sum() == origins.shape[0]:
                break

            t += distance_to_surface

        return points, mask

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]
            points, mask = self.sphere_tracing(
                implicit_fn,
                cur_ray_bundle.origins,
                cur_ray_bundle.directions
            )
            mask = mask.repeat(1,3)
            isect_points = points[mask].view(-1, 3)

            # Get color from implicit function with intersection points
            isect_color = implicit_fn.get_color(isect_points)

            # Return
            color = torch.zeros_like(cur_ray_bundle.origins)
            color[mask] = isect_color.view(-1)

            cur_out = {
                'color': color.view(-1, 3),
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


def sdf_to_density(signed_distance, alpha, beta):
    # TODO (Q7): Convert signed distance to density with alpha, beta parameters
    """
    signed_distance.shape = (N, 1)
    alpha.shape = (N, 1)
    beta.shape = (N, 1)
    """
    # signed_distance = -signed_distance
    # density = torch.where(
    #     signed_distance < 0,
    #     0.5 * torch.exp(signed_distance / beta),
    #     1 - 0.5 * torch.exp(-signed_distance / beta)
    # )
    lap_dist = torch.distributions.laplace.Laplace(0, beta)

    return alpha * lap_dist.cdf(-signed_distance)

    # return alpha * density

class VolumeSDFRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.cfg = cfg

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(cur_ray_bundle.sample_points)
            density = sdf_to_density(distance, self.alpha, self.beta)

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            )

            geometry_color = torch.zeros_like(color)

            # Compute color
            color = self._aggregate(
                weights,
                color.view(-1, n_pts, color.shape[-1])
            )

            # Return
            cur_out = {
                'color': color,
                "geometry": geometry_color
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer,
    'sphere_tracing': SphereTracingRenderer,
    'volume_sdf': VolumeSDFRenderer
}
