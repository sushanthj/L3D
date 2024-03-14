import torch
import torch.nn.functional as F

def eikonal_loss(gradients):
    # gradients.shape = (N, 3)
    # ensure that all gradients have a magnitude of 1
    # to do so, we enfore a mean square difference loss, where the difference = (norm(true_grad) - desired_grad)
    # if desired grad = 1
    # then the loss = ((norm(true_grad) - 1)^2).mean()
    return torch.mean(torch.square(torch.abs(torch.norm(gradients, dim=-1)- 1.)))

def sphere_loss(signed_distance, points, radius=1.0):
    return torch.square(signed_distance[..., 0] - (torch.norm(points, dim=-1) - radius)).mean()

def get_random_points(num_points, bounds, device):
    min_bound = torch.tensor(bounds[0], device=device).unsqueeze(0)
    max_bound = torch.tensor(bounds[1], device=device).unsqueeze(0)

    return torch.rand((num_points, 3), device=device) * (max_bound - min_bound) + min_bound

def select_random_points(points, n_points):
    points_sub = points[torch.randperm(points.shape[0])]
    return points_sub.reshape(-1, 3)[:n_points]
