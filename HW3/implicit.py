import torch
import torch.nn.functional as F
from torch import autograd

from ray_utils import RayBundle


# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.radius = torch.nn.Parameter(torch.tensor(cfg.radius.val).float(), requires_grad=cfg.radius.opt)
        self.center = torch.nn.Parameter(torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt)

    def forward(self, points):
        points = points.view(-1, 3)

        return torch.linalg.norm(points - self.center, dim=-1, keepdim=True) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt)
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(cfg.side_lengths.val).float().unsqueeze(0), requires_grad=cfg.side_lengths.opt)
        """
        example for box:
            self.center = torch.nn.Parameter([0.0, 0.0, 0.0].unsqueeze(0), requires_grad=True)
            self.side_lengths = torch.nn.Parameter([1.75, 1.75, 1.75].unsqueeze(0), requires_grad=True)
        """

    def forward(self, points):
        """
        Here we'll get the SDF value at the queried points using our above self.center and self.side_lengths
        """
        points = points.view(-1, 3)
        diff = torch.abs(points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)

# Torus SDF class
class TorusSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.radii = torch.nn.Parameter(
            torch.tensor(cfg.radii.val).float().unsqueeze(0), requires_grad=cfg.radii.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = points - self.center
        q = torch.stack(
            [
                torch.linalg.norm(diff[..., :2], dim=-1) - self.radii[..., 0],
                diff[..., -1],
            ],
            dim=-1
        )
        return (torch.linalg.norm(q, dim=-1) - self.radii[..., 1]).unsqueeze(-1)

sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
    'torus': TorusSDF,
}


# Converts SDF into density/feature volume
class SDFVolume(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )

        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )

        self.alpha = torch.nn.Parameter(
            torch.tensor(cfg.alpha.val).float(), requires_grad=cfg.alpha.opt
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(cfg.beta.val).float(), requires_grad=cfg.beta.opt
        )

    def _sdf_to_density(self, signed_distance):
        # Convert signed distance to density with alpha, beta parameters
        return torch.where(
            signed_distance > 0,
            0.5 * torch.exp(-signed_distance / self.beta),
            1 - 0.5 * torch.exp(signed_distance / self.beta),
        ) * self.alpha

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)
        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        ).view(-1, 1)

        # Transform SDF to density
        signed_distance = self.sdf(ray_bundle.sample_points)
        density = self._sdf_to_density(signed_distance)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(sample_points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        out = {
            'density': -torch.log(1.0 - density) / deltas,
            'feature': base_color * self.feature * density.new_ones(sample_points.shape[0], 1)
        }

        return out


# Converts SDF into density/feature volume
class SDFSurface(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )
        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )
    
    def get_distance(self, points):
        points = points.view(-1, 3)
        return self.sdf(points)

    def get_color(self, points):
        points = points.view(-1, 3)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        return base_color * self.feature * points.new_ones(points.shape[0], 1)
    
    def forward(self, points):
        return self.get_distance(points)

class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        # output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        """

        Args:
            n_layers (int): number of layers in MLP
            input_dim (int): input dimension at first layer of MLP
            skip_dim (int):  # dimension of skip connection which gets added to layer: in our case it's the input_dim
            hidden_dim (int): # output size of each layer
            input_skips (List): At which layer the input will be added to a layer's output as skip connection
        """
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

        # xavier init the weights
        for layer in self.mlp:
            torch.nn.init.xavier_uniform_(layer[0].weight)
            torch.nn.init.zeros_(layer[0].bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


# TODO (Q3.1): Implement NeRF MLP without View Dependence
class NeuralRadianceField_without_view(torch.nn.Module):
    def __init__(self, cfg):
        """
        Architecture for NeRF MLP without View Dependence

              -> |  -> |  -> |  -> |  -> |  -> |  -> |  -> |
        input -> |  -> |  -> |  -> |  -> |  -> |  -> |  -> | -> Linear(hidden_dim, 4) -> output
          |   -> |  -> |  -> |  -> |  -> |  -> |  -> |  -> |
          |                        |
          └──-----skip con---------'


        Relu[output[..., 0]] = Density prediciton
        Sigmoid[output[..., 1:4]] = Feature/Color prediction

        """
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim

        # MLP for xyz (no view dependence) equivalent to constructing upto 8th layer of MLP shown in NERF paper
        self.MLP = MLPWithInputSkips(
            n_layers = cfg.n_layers_xyz, # number of layers in MLP
            input_dim = embedding_dim_xyz, # the Harmonic embedding layer between the input and the 1st hidden layer of MLP
            # output_dim = None, # seems to not be used in MLPWithInputSkips
            skip_dim = embedding_dim_xyz, # dimension of skip connection which gets added to layer: in our case it's the input_dim
            hidden_dim = cfg.n_hidden_neurons_xyz, # output size of each layer
            input_skips = [4], # list of layers where skip connection is added eg. [4] as per NERF paper
        )
        self.final_linear = torch.nn.Linear(cfg.n_hidden_neurons_xyz, 4) # as cfg.n_hidden_neurons_xyz = hidden_dim

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    """
    Your MLP should take in a RayBundle object in its forward method, and produce color and density for each sample point in the RayBundle.
    """
    def forward(self, ray_bundle):
        embedding_xyz = self.harmonic_embedding_xyz(ray_bundle.sample_points.view(-1, 3))
        x = self.MLP(x=embedding_xyz, z=embedding_xyz) # x = input, z = skip connection
        x = self.final_linear(x)
        density = self.relu(x[:, 0].unsqueeze(-1))
        feature = self.sigmoid(x[:, 1:])
        out = {'density': density, 'feature': feature}

        return out



# TODO (Q4.1): Implement NeRF MLP with View Dependence
class NeuralRadianceField(torch.nn.Module):
    def __init__(self, cfg):
        """
        Architecture for NeRF MLP without View Dependence

              -> |  -> |  -> |  -> |  -> |  -> |  -> |  -> |
        input -> |  -> |  -> |  -> |  -> |  -> |  -> |  -> | -> Linear(hidden_dim, hidden_dim+1) -> output
          |   -> |  -> |  -> |  -> |  -> |  -> |  -> |  -> |
          |                        |
          └──-----skip con---------'


        - Density prediciton = Relu[output[..., 0]]

        - concat_dir_xyz = torch.cat((embedding_dir, x[:, 1:]), dim=1)
        - feature_pred = MLP(concat_dir_xyz)
        - feature = Linear(feature_pred, 3)

        - Feature/Color prediction = Sigmoid[feature]
        """
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim

        # MLP for xyz (no view dependence) equivalent to constructing upto 8th layer of MLP shown in NERF paper
        self.xyz_MLP = MLPWithInputSkips(
            n_layers = cfg.n_layers_xyz, # number of layers in MLP
            input_dim = embedding_dim_xyz, # the Harmonic embedding layer between the input and the 1st hidden layer of MLP
            # output_dim = None, # seems to not be used in MLPWithInputSkips
            skip_dim = embedding_dim_xyz, # the layer which gets used as skip connection
            hidden_dim = cfg.n_hidden_neurons_xyz, # output size of each layer
            input_skips = [4], # list of layers where skip connection is added eg. [4] as per NERF paper
        )

        self.feature_MLP = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim_dir+cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_dir),
                torch.nn.ReLU(),
                torch.nn.Linear(cfg.n_hidden_neurons_dir, 3),
                torch.nn.Sigmoid()
            )

        self.xyz_to_sigma = torch.nn.Sequential(
                torch.nn.Linear(cfg.n_hidden_neurons_xyz, 1),
                torch.nn.ReLU()
            )

        self.xyz_to_feature_transition = torch.nn.Sequential(
                torch.nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_xyz),
                torch.nn.ReLU()
            )


    def forward(self, ray_bundle):
        # xyz part of network
        embedding_xyz = self.harmonic_embedding_xyz(ray_bundle.sample_points.view(-1, 3))
        x = self.xyz_MLP(x=embedding_xyz, z=embedding_xyz)

        # network splits into two parts here
        density = self.xyz_to_sigma(x)
        feature = self.xyz_to_feature_transition(x)

        # making the transition
        embedding_dir = self.harmonic_embedding_dir(ray_bundle.directions).unsqueeze(1)
        embedding_dir = torch.tile(embedding_dir, (1, feature.shape[1], 1)).view(-1, embedding_dir.shape[-1])
        # embedding_dir = embedding_dir.repeat_interleave(feature.shape[1], dim=1).view(-1, embedding_dir.shape[-1])
        concat_xyz_and_dir = torch.cat((embedding_dir, feature), dim=-1)
        feature = self.feature_MLP(concat_xyz_and_dir)

        out = {'density': density, 'feature': feature}

        return out


class NeuralSurface(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # TODO (Q6): Implement Neural Surface MLP to output per-point SDF
        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim

        # MLP for xyz (no view dependence) equivalent to constructing upto 8th layer of MLP shown in NERF paper
        self.MLP_dist_or_color = MLPWithInputSkips(
            n_layers = cfg.n_layers_distance, # number of layers in MLP
            input_dim = embedding_dim_xyz, # the Harmonic embedding layer between the input and the 1st hidden layer of MLP
            # output_dim = None, # seems to not be used in MLPWithInputSkips
            skip_dim = embedding_dim_xyz, # dimension of skip connection which gets added to layer: in our case it's the input_dim
            hidden_dim = cfg.n_hidden_neurons_distance, # output size of each layer
            input_skips = [4], # list of layers where skip connection is added eg. [4] as per NERF paper
        )

        self.final_linear_dist = torch.nn.Linear(cfg.n_hidden_neurons_distance, 1) # as cfg.n_hidden_neurons_xyz = hidden_dim

        # TODO (Q7): Implement Neural Surface MLP to output per-point color

        # MLP for xyz (no view dependence) equivalent to constructing upto 8th layer of MLP shown in NERF paper
        self.MLP_combined = MLPWithInputSkips(
            n_layers = cfg.n_layers_color, # number of layers in MLP
            input_dim = cfg.n_hidden_neurons_distance, # the Harmonic embedding layer between the input and the 1st hidden layer of MLP
            # output_dim = None, # seems to not be used in MLPWithInputSkips
            skip_dim = 0, # dimension of skip connection which gets added to layer: in our case it's the input_dim
            hidden_dim = cfg.n_hidden_neurons_color, # output size of each layer
            input_skips = [], # list of layers where skip connection is added eg. [4] as per NERF paper
        )

        self.final_linear_color = torch.nn.Linear(cfg.n_hidden_neurons_color, 3)
        self.sigmoid = torch.nn.Sigmoid()

    def get_distance(
        self,
        points
    ):
        '''
        TODO: Q6
        Output:
            distance: N X 1 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)
        embedding_xyz = self.harmonic_embedding_xyz(points)
        x = self.MLP_dist_or_color(x=embedding_xyz, z=embedding_xyz)
        x = self.final_linear_dist(x)
        return x

    def get_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Output:
            distance: N X 3 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)
        # pass
        embedding_xyz = self.harmonic_embedding_xyz(points)
        x = self.MLP_dist_or_color(x=embedding_xyz, z=embedding_xyz)
        x = self.MLP_combined(x=x, z=x)
        x = self.final_linear_color(x)
        x = self.sigmoid(x)
        return x

    def get_distance_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Output:
            distance, points: N X 1, N X 3 Tensors, where N is number of input points
        You may just implement this by independent calls to get_distance, get_color
            but, depending on your MLP implementation, it maybe more efficient to share some computation
        '''
        points = points.view(-1, 3)
        embedding_xyz = self.harmonic_embedding_xyz(points)
        x = self.MLP_dist_or_color(x=embedding_xyz, z=embedding_xyz)
        distance = self.final_linear_dist(x)

        x = self.MLP_combined(x=x, z=x)
        color = self.final_linear_color(x)
        color = self.sigmoid(color)

        return distance, color

    def forward(self, points):
        return self.get_distance(points)

    def get_distance_and_gradient(
        self,
        points
    ):
        has_grad = torch.is_grad_enabled()
        points = points.view(-1, 3)

        # Calculate gradient with respect to points
        with torch.enable_grad():
            points = points.requires_grad_(True)
            distance = self.get_distance(points)
            gradient = autograd.grad(
                distance,
                points,
                torch.ones_like(distance, device=points.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]

        return distance, gradient


implicit_dict = {
    'sdf_volume': SDFVolume,
    'nerf': NeuralRadianceField,
    'sdf_surface': SDFSurface,
    'neural_surface': NeuralSurface,
    'nerf_without_view': NeuralRadianceField_without_view,
}
