import argparse
import time

import dataset_location
import losses
import torch
import pickle
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes
from r2n2_custom_fast import R2N2


def get_args_parser():
    parser = argparse.ArgumentParser("Singleto3D", add_help=False)
    # Model parameters
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--max_iter", default=100000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--type", default="vox", choices=["vox", "point", "mesh"], type=str)
    parser.add_argument("--n_points", default=1000, type=int)
    parser.add_argument("--w_chamfer", default=1.0, type=float)
    parser.add_argument("--w_smooth", default=0.1, type=float)
    parser.add_argument("--save_freq", default=2000, type=int)
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument('--load_feat', action='store_true')
    parser.add_argument("--device", default="cuda", type=str)
    return parser


def preprocess(feed_dict, args):
    images = feed_dict["images"].squeeze(1)
    if args.type == "vox":
        voxels = feed_dict["voxels"].float()
        ground_truth_3d = voxels
    elif args.type == "point":
        mesh = feed_dict["mesh"]
        pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)
        ground_truth_3d = pointclouds_tgt
    elif args.type == "mesh":
        ground_truth_3d = feed_dict["mesh"]
    if args.load_feat:
        feats = torch.stack(feed_dict["feats"])
        return feats.to(args.device), ground_truth_3d.to(args.device)
    else:
        return images.to(args.device), ground_truth_3d.to(args.device)


def calculate_loss(predictions, ground_truth, args):
    if args.type == "vox":
        loss = losses.voxel_loss(predictions, ground_truth)
    elif args.type == "point":
        loss = losses.chamfer_loss(predictions, ground_truth)
    elif args.type == "mesh":
        sample_trg = sample_points_from_meshes(ground_truth, args.n_points)
        sample_pred = sample_points_from_meshes(predictions, args.n_points)

        loss_reg = losses.chamfer_loss(sample_pred, sample_trg)
        loss_smooth = losses.smoothness_loss(predictions)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth
    return loss


def train_model(args):
    r2n2_dataset = R2N2(
        "train",
        dataset_location.SHAPENET_PATH,
        dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH,
        return_voxels=False,
        return_feats=args.load_feat,
        use_cache=False,
    )

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=collate_batched_R2N2,
        pin_memory=False,
        drop_last=True,
        shuffle=False,
    )

    train_loader = iter(loader)

    print("Loading training data !")
    max_iter = len(train_loader)
    epoch = -1
    start_time = time.time()

    for step in range(0, max_iter):
        iter_start_time = time.time()

        if step % len(train_loader) == 0:  # restart after one epoch
            train_loader = iter(loader)
            epoch+=1

        read_start_time = time.time()

        feed_dict = next(train_loader)
        # images_gt, ground_truth_3d = preprocess(feed_dict, args)
        read_time = time.time() - read_start_time
        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time
        print("Step: ", step)

    # print("Pickling data")
    # print(len(loader.dataset.mesh_cache.keys()))
    loader.dataset._save_mesh_pickle()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
