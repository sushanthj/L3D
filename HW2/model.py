from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class View(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape

        def forward(self, x):
            return x.view(*self.shape)

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

# NOTE: For visualizing the voxels, make the voxel size proportional to the confidence of the output
#       See Fig. 4 in Learning a Predictable and Generative Vector Representation for Objects

        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            self.voxel_decode = self.voxel_decoder()
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3
            self.n_point = args.n_points
            self.point_decode = self.point_decoder()
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            self.mesh_decode = self.mesh_decoder()

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            voxels_pred = self.voxel_decode(encoded_feat)
            return voxels_pred

        elif args.type == "point":
            pointclouds_pred = self.point_decode(encoded_feat)
            return pointclouds_pred

        elif args.type == "mesh":
            deform_vertices_pred = self.mesh_decode(encoded_feat)
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred

    def voxel_decoder(self):
        """
        Define the Decoder to generate voxels from the base latent vector (b x 512)

        From Pix2View
        """
        decoder = nn.Sequential(
            nn.Linear(512, 256*4*4*4),  # Map the input features to the volume of the voxel grid
            View((-1, 256, 4, 4, 4)),  # Reshape the tensor to the right size
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output probabilities between 0 and 1
        )
        return decoder


    def point_decoder(self):
        """
        Define the Decoder to generate point clouds from the base latent vector (b x 512)
        """
        decoder = nn.Sequential(
            nn.Linear(512, 1024),  # Map the input features to the volume of the voxel grid
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_point*3),
            # split the output into (b x n_point x 3)
            View((-1, self.n_point, 3))
        )
        return decoder


    def mesh_decoder(self):
        """
        Define the Decoder to generate meshes from the base latent vector (b x 512)
        """
        decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.mesh_pred.verts_packed().shape[0]*3),
            # split the output into b x mesh_pred.verts_packed().shape[0] x 3
            View((-1, self.mesh_pred.verts_packed().shape[0], 3))
        )
        return decoder