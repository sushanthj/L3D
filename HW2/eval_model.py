import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt
from visualie_utils import *
import torch.nn.functional as F

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--vis_freq', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=3000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--load_checkpoint', action='store_true')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--load_feat', action='store_true')
    parser.add_argument('--checkpoint_path', default='./checkpoints/point_large_new_arch_n3000.pth', type=str)
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh

def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    plt.savefig(f'eval_{args.type}', bbox_inches='tight')


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = predictions
        H,W,D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
        # if vertices_src.shape == (0,3) and faces_src.shape == (0,3):
            # return
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)

    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics

# Define the hook function
def hook(module, input, output_tensor):
    global output
    output = output_tensor

@torch.inference_mode()
def evaluate_model(args):
    r2n2_dataset = R2N2("test",
                        dataset_location.SHAPENET_PATH,
                        dataset_location.R2N2_PATH,
                        dataset_location.SPLITS_PATH,
                        return_voxels=True,
                        return_feats=args.load_feat,)
                        # use_cache=True,)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    intermediate_tensors_for_viz = []

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    if args.load_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")

    print("Starting evaluating !")
    max_iter = len(eval_loader)
    skipped = 0

    # handle = model.mesh_decode.linear1.register_forward_hook(hook)

    for step in range(start_iter, max_iter):
        # if (step > 100) and ((step % 230 == 0) or (step % 520 == 0) or (step % 600 == 0) or (step % 540 == 0) or (step % 490 == 0)):
        #     pass
        # else:
        #     continue

        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)

        if args.type == "vox":
            predictions = predictions.permute(0,1,4,3,2)

        metrics = evaluate(predictions, mesh_gt, thresholds, args)

        if (step % args.vis_freq) == 0:
            # visualization block
            # visualize_mesh(mesh_gt, f'{step}_gt')

            if args.type == "vox":
                try:
                    visualize_voxels_as_mesh(predictions, f'{step}_{args.type}')
                except:
                    print("decrease ISO")
                    skipped += 1
            elif args.type == "point":
                # visualize_pointcloud(predictions, f'{step}_{args.type}')
                pass
                # visualize_point_difference(predictions, mesh_gt, f'diff_{step}_{args.type}')
            elif args.type == "mesh":
                visualize_mesh(predictions, f'{step}_{args.type}')
                # interpolate the best looking objects
                # if step > 100:
                #     if (step % 220 == 0) or (step % 230  == 0) or (step % 480 == 0) or (step % 540 == 0) or (step % 490 == 0):
                #         # add latent tensor for future visualization
                #         intermediate_tensors_for_viz.append(output)
                #         visualize_mesh(predictions, f'pred_{step}_{args.type}')
                #         visualize_mesh(mesh_gt, f'{step}_gt')
            else:
                raise ValueError(f"Unknown type: {args.type}")
            # plt.imsave(f'vis/{step}_{args.type}.png', rend)


        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        # metrics at threshold = 0.05
        f1_05 = metrics['F1@0.050000']
        avg_f1_score_05.append(f1_05)
        avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score,  args)
    print('Done!')
    print("skipped", skipped)

    # interpolated_tesors = []
    # for i in range(len(intermediate_tensors_for_viz)-1):
    #     for j in range(5):
    #         left_weight = j*2*0.1
    #         right_weight = 1 - left_weight
    #         interpolated = left_weight * intermediate_tensors_for_viz[i] + right_weight * intermediate_tensors_for_viz[i+1]
    #         interpolated_tesors.append(interpolated)

    # handle.remove()

    # img_list = []
    # # Now pass each individual tensor in interpolated_tesors through the model and visualize the output
    # for i, tensor in enumerate(interpolated_tesors):
    #     predictions = model(images_gt, args, intermediate_output=tensor)
    #     img_list.append(render_mesh_to_img(predictions, f'interpolated_{i}_{args.type}'))

"""
NOTE: TA comment on Piazza:
'There is no exact number to hit. However, as a reference,
- F1 score > 65 for voxels
- F1 score > 90 for point cloud and mesh

should be a good starting point to reach
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
