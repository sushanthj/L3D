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
            # NOTE: for a ico_sphere the value of mesh_pred.verts_packed().shape[0] = 652
            mesh_pred = ico_sphere(4, self.device)
            self.n_verts = mesh_pred.verts_packed().shape[0]
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            print(" pre model shape ", self.mesh_pred.verts_packed().shape)
            n_verts = self.n_verts
            self.mesh_decode = MeshDecoder(n_verts)

    def forward(self, images, args, intermediate_output=None):
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
            # print("post model shape ", (deform_vertices_pred.reshape([-1,3])).shape)
            # print(" pre model shape ", self.mesh_pred.verts_packed().shape)
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3])) #! ASK: Are we collapsing all batches?
            return  mesh_pred


    def voxel_decoder(self):
        """
        Define the Decoder to generate voxels from the base latent vector (b x 512)

        From Pix2View
        """
        decoder = nn.Sequential(
            nn.Linear(512,2048),  # Map the input features to the volume of the voxel grid
            # View((-1, 256, 2, 2, 2)),  # Reshape the tensor to the right size
            nn.Unflatten(1,(256,2,2,2)),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.ConvTranspose3d(256, 384, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm3d(384),
            nn.ReLU(),
            nn.ConvTranspose3d(384, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.2),
            # NOTE: the output of the last layer should be a 3D volume of size 32x32x32
            # Here, we reduce the number of channels to 1 to get output shape = (b x 1 x 32 x 32 x 32)
            nn.ConvTranspose3d(256, 96, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose3d(96, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()  # Output probabilities between 0 and 1
        )
        return decoder


    def point_decoder(self):
        """
        Define the Decoder to generate point clouds from the base latent vector (b x 512)
        """
        decoder = nn.Sequential(
            nn.Linear(512, 2048),  # Map the input features to the volume of the voxel grid
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 3046),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(3046, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, self.n_point*3),
            # split the output into (b x n_point x 3)
            View((-1, self.n_point, 3))
        )
        return decoder


class MeshDecoder(nn.Module):
    def __init__(self, n_verts):
        super().__init__()
        self.n_verts = n_verts
        self.linear1 = nn.Linear(512, 2048)
        self.gelu1 = nn.GELU()
        self.drop1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(2048, 2048)
        self.gelu2 = nn.GELU()
        self.linear3 = nn.Linear(2048, self.n_verts*3)
        self.tanh = nn.Tanh()

    def forward(self, img, intermediate_output=None):
        if intermediate_output is None:
            x = self.linear1(img)
        else:
            x = intermediate_output
        x = self.gelu1(x)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.gelu2(x)
        x = self.linear3(x)
        x = self.tanh(x)
        return x