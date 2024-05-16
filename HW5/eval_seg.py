import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model().to(args.device)

    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data_full = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label_full = torch.from_numpy((np.load(args.test_label))[:,ind])

    # # ------ TO DO: Make Prediction ------
    # pred_label = model(test_data.to(args.device)) # will be of shape Nx10000x3
    # pred_label = torch.argmax(pred_label, dim=-1)

    # test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    # print ("test accuracy: {}".format(test_accuracy))
    
    # max_len = test_label.shape[0]
    # randints = np.random.randint(0, high=max_len, size=100, dtype=int)

    # for i in randints:
    #     # Visualize Segmentation Result (Pred VS Ground Truth)
    #     viz_seg(test_data[i], test_label[i], "{}/gt_{}.gif".format(args.output_dir, args.exp_name), args.device)
    #     viz_seg(test_data[i], pred_label[i], "{}/pred_{}.gif".format(args.output_dir, args.exp_name), args.device)

    batch_size = 200
    num_batches = np.ceil(test_data_full.shape[0]/batch_size)
    cumulative_acc = 0
    print("num batches is", num_batches)
    for i in range(0, test_data_full.shape[0], batch_size):
        test_data = test_data_full[i:i+200]
        test_label = test_label_full[i:i+200]
        # ------ TO DO: Make Prediction ------
        pred_label = model(test_data.to(args.device)) # will be of shape Nx10000x3 (3 classes)

        # Compute Accuracy
        pred_label = torch.argmax(pred_label, dim=-1) # get the most likely class -> Nx10000x1
        test_label = test_label.to(args.device)
        test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
        cumulative_acc += test_accuracy
        print ("test accuracy: {}".format(test_accuracy))
        
        max_len = test_label.shape[0]
        randints = np.random.randint(0, high=max_len, size=100, dtype=int)

        for i in randints:
            # Visualize Segmentation Result (Pred VS Ground Truth)
            viz_seg(test_data[i], test_label[i], "{}/{}_gt.gif".format(args.output_dir, i), args.device)
            viz_seg(test_data[i], pred_label[i], "{}/{}_pred.gif".format(args.output_dir, i), args.device)
        
    print(f"Done! Average Segmentation accuracy was {cumulative_acc/num_batches}")