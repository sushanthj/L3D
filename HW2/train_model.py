import argparse
import time

import dataset_location
import losses
import torch
import wandb
from model import SingleViewto3D
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes
from r2n2_custom import R2N2


def get_args_parser():
    parser = argparse.ArgumentParser("Singleto3D", add_help=False)
    # Model parameters
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--max_iter", default=20000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--type", default="vox", choices=["vox", "point", "mesh"], type=str
    )
    parser.add_argument("--n_points", default=5000, type=int)
    parser.add_argument("--w_chamfer", default=1.0, type=float)
    parser.add_argument("--w_smooth", default=1.2, type=float)
    parser.add_argument("--save_freq", default=500, type=int)
    parser.add_argument("--load_checkpoint", default=False)
    parser.add_argument('--load_feat', action='store_true')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--wandb_run_name', default='voxel_2', type=str)
    parser.add_argument('--return_voxels', default=False, type=bool)
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
        return_voxels=args.return_voxels, # set to False for point cloud and mesh
        return_feats=args.load_feat,
    )

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    train_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.train()

    # checkpoint_path = '/content/drive/MyDrive/Colab Notebooks/L3D/Assignment_1/checkpoints/voxel_2.pth'
    # checkpoint_path = '/home/sush/CMU/l3d/L3D/HW2/checkpoints/voxel_1.pth'
    # checkpoint_path = f'/home/mrsd_teamh/sush/L3D/HW2/checkpoints/{args.type}_1.pth'
    checkpoint_path = f'/projects/academic/rohini/m44/git-prjs/3DVision/L3D/HW2/checkpoint_{args.type}.pth'
    wandb.login(key="49efd84d0e342f343fb91401332234dea4a3ffe2")

    config = {
        "arch": args.arch,
        "lr": args.lr,
        "max_iter": args.max_iter,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "type": args.type,
        "n_points": args.n_points,
        "w_chamfer": args.w_chamfer,
        "w_smooth": args.w_smooth,
        "save_freq": args.save_freq,
        "load_checkpoint": args.load_checkpoint,
        "load_feat": args.load_feat,
    }

    # Create your wandb run
    run = wandb.init(
        name    = args.wandb_run_name, ### Wandb creates random run names if you skip this field, we recommend you give useful names
        reinit  = True, ### Allows reinitalizing runs when you re-run this cell
        #id     =
        #resume =
        project = "L3D Assignment_1", ### Project should be created in your wandb account
        config  = config ### Wandb Config for your run
    )

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # to use with ViTs
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[26000], gamma=0.5)
    start_iter = 0
    start_time = time.time()

    if args.load_checkpoint:
        # checkpoint = torch.load(f"checkpoint_{args.type}.pth")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # start_iter = checkpoint["step"]
        print(f"Succesfully loaded iter {start_iter}")

    min_loss = 1000000

    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        if step % len(train_loader) == 0:  # restart after one epoch
            train_loader = iter(loader)

        read_start_time = time.time()

        feed_dict = next(train_loader)

        images_gt, ground_truth_3d = preprocess(feed_dict, args)
        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt, args)

        loss = calculate_loss(prediction_3d, ground_truth_3d, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        if (step % args.save_freq) == 0 and step > 0:
            if loss_vis < min_loss:
                print(f"Saving checkpoint at step {step}")
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    f"checkpoint_{args.type}.pth",
                )
                checkpoint = {'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'Train loss': loss,
                }
                checkpoint_path = checkpoint_path = f'/projects/academic/rohini/m44/git-prjs/3DVision/L3D/HW2/checkpoints/{args.type}_{step}.pth'
                torch.save(checkpoint, checkpoint_path)
                min_loss = loss_vis

            wandb.log({'train_loss': loss, 'lr': scheduler._last_lr[0]})

        # step is not the same as epoch, call scheduler only at the end of each epoch
        # if step % len(train_loader) == 0:
        scheduler.step()

        print(
            "[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f"
            % (step, args.max_iter, total_time, read_time, iter_time, loss_vis)
        )

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
