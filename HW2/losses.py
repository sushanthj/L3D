import torch
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.loss import mesh_laplacian_smoothing, chamfer_distance

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	"""
	NOTE: BCE is CrossEntropy that's meant to be used for binary classification

	#? In CE Loss, the output is a probability distribution across all classes (softmax)
	CELoss = - summation(logsoftmax(prediction_vector) * label)

	#? In BCE Loss, the output for each sample in a batch is a scalar -> which is typically
	#? passed through a sigmoid to get the binary prediction value (0 or 1). 
	#? This is then compared to the label (also 0 or 1) as:
	BCELoss = -[label *log{sigmoid(prediciton)} + (1-label)*log{1-sigmoid(prediction)}] -> Basically KL Divergence

	Now, BCEWithLogitsLoss is a small modification where it internally does the sigmoid for us
	"""
	loss_func = torch.nn.BCELoss()
	loss = loss_func(voxel_src, voxel_tgt)
	# implement some loss for binary voxel grids
	return loss


# NOTE: chamfer loss is only used with pointclouds, with voxels BCE is sufficient
def chamfer_loss(point_cloud_src, point_cloud_tgt):
	# point_cloud_src, point_cloud_tgt: b x n_points x 3

	# NOTE: below line will yield 1 point in cloud target which is closest to the same index pt of src
	# forward_result = knn_points(point_cloud_src, point_cloud_tgt, K=1) # output shape = b x n_points x 1
	# sqaured_distances, idx = forward_result
	# # we can then use gather to get the actual point coordinates in cloud target
	# forward_gather = knn_gather(point_cloud_tgt, forward_result.idx, forward_result.shape[-1])

	# For numerical stability in sq root
	# eps = 1e-8

	# knn_result = knn_points(point_cloud_src, point_cloud_tgt, K=1) # output shape = b x n_points x 1
	# loss_part_1 = (torch.sqrt(knn_result.dists + eps)).mean()

	# # Repeat for backward
	# knn_result = knn_points(point_cloud_tgt, point_cloud_src, K=1) # output shape = b x n_points x 1
	# loss_part_2 = (torch.sqrt(knn_result.dists + eps)).mean()

	# return (loss_part_1 + loss_part_2)

	loss, _ = chamfer_distance(point_cloud_src, point_cloud_tgt)
	return loss

def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	# implement laplacian smoothening loss
	return loss_laplacian