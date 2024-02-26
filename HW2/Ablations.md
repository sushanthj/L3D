# Pix to Vox training

## Using Resnet18, No batchnorm, not training backbone

- 4k iterations
- vox.pth
- ```python
  decoder = nn.Sequential(
            nn.Linear(512, 256*4*4*4),  # Map the input features to the volume of the voxel grid
            View((-1, 256, 4, 4, 4)),  # Reshape the tensor to the right size
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # NOTE: the output of the last layer should be a 3D volume of size 32x32x32
            # Here, we reduce the number of channels to 1 to get output shape = (b x 1 x 32 x 32 x 32)
            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output probabilities between 0 and 1
        )
  ```

  This had evaluation metric:


## Using Resnet18 and Batchnorm, training backbone

- 25k iterations
- vox_1.pth
- ```python
  decoder = nn.Sequential(
            nn.Linear(512, 128*4*4*4),  # Map the input features to the volume of the voxel grid
            View((-1, 128, 4, 4, 4)),  # Reshape the tensor to the right size
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # NOTE: the output of the last layer should be a 3D volume of size 32x32x32
            # Here, we reduce the number of channels to 1 to get output shape = (b x 1 x 32 x 32 x 32)
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output probabilities between 0 and 1
        )
        return decode
  ```

- This model had an evaluation metric:


## Using Resnet18 and GELU and 1 extra layer

- different number of iterations
- vox_2.pth
```python
decoder = nn.Sequential(
            nn.Linear(512,2048),  # Map the input features to the volume of the voxel grid
            View((-1, 256, 2, 2, 2)),  # Reshape the tensor to the right size
            nn.GELU(),
            nn.ConvTranspose3d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.GELU(),
            nn.ConvTranspose3d(256, 384, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(384),
            nn.GELU(),
            nn.ConvTranspose3d(384, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GELU(),
            # NOTE: the output of the last layer should be a 3D volume of size 32x32x32
            # Here, we reduce the number of channels to 1 to get output shape = (b x 1 x 32 x 32 x 32)
            nn.ConvTranspose3d(256, 96, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.ConvTranspose3d(96, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()  # Output probabilities between 0 and 1
          )
```

## v3 (vox_3.pth)

```python
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
```






# Image to Pointcloud

## Without Batchnorm (vanilla) without training backbone

- 12k iterations
- point.pth
```python
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
```

<!-- insert image -->

## With Batchnorm and w_smooth = 0.1 and not training backbone

- 50k iterations
- point_2.pth
```python
decoder = nn.Sequential(
    nn.Linear(512, 1024),  # Map the input features to the volume of the voxel grid
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Linear(1024, self.n_point*3),
    # split the output into (b x n_point x 3)
    View((-1, self.n_point, 3))
)
return decoder
```

## With Batchnorm and training backbone and GELU instead of ReLU

- 45k iterations
- point_3.pth
```python
decoder = nn.Sequential(
            nn.Linear(512, 1024),  # Map the input features to the volume of the voxel grid
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, self.n_point*3),
            # split the output into (b x n_point x 3)
            View((-1, self.n_point, 3))
        )
        return decoder
```

## Without Bachnorm (point_3_xx.pth)

```python
def point_decoder(self):
        """
        Define the Decoder to generate point clouds from the base latent vector (b x 512)
        """
        decoder = nn.Sequential(
            nn.Linear(512, 1024),  # Map the input features to the volume of the voxel grid
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_point*3),
            # split the output into (b x n_point x 3)
            View((-1, self.n_point, 3))
        )
        return decoder
```

## Large Model (old arch) - ~12000 iterations

```python
decoder = nn.Sequential(
            nn.Linear(512, 1024),  # Map the input features to the volume of the voxel grid
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, self.n_point*3),
            # split the output into (b x n_point x 3)
            View((-1, self.n_point, 3))
        )
```


## Large Model Training (new arch) ~ 

```python
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
```


# Image to Mesh

## v1

```python
decoder = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, self.n_verts*3),
            nn.Tanh() # we want -1 to 1 output
            # NOTE: self.mesh_pred.verts_packed().shape[0] includes the batch size
            # split the output into b x mesh_pred.verts_packed().shape[0] x 3
            # View((-1, self.mesh_pred.verts_packed().shape[0], 3))
        )
```

## v2

```python
decoder = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, self.n_verts*3),
            nn.Tanh() # we want -1 to 1 output
            # NOTE: self.mesh_pred.verts_packed().shape[0] includes the batch size
            # split the output into b x mesh_pred.verts_packed().shape[0] x 3
            # View((-1, self.mesh_pred.verts_packed().shape[0], 3))
        )
```

## v3 - RELU

```python
class MeshDecoder(nn.Module):
    def __init__(self, n_verts):
        super().__init__()
        self.n_verts = n_verts
        self.linear1 = nn.Linear(512, 2048)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(2048, 2048)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(2048, self.n_verts*3)
        self.tanh = nn.Tanh()

    def forward(self, x, intermediate_output=None):
        if intermediate_output is None:
            x = self.linear1(x)
        else:
            x = intermediate_output
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.tanh(x)
        return x
```

# v4 - GELU

```python
class MeshDecoder(nn.Module):
    def __init__(self, n_verts):
        super().__init__()
        self.n_verts = n_verts
        self.linear1 = nn.Linear(512, 2048)
        self.gelu1 = nn.GELU()
        self.drop1 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(2048, 2048)
        self.gelu2 = nn.GELU()
        self.linear3 = nn.Linear(2048, self.n_verts*3)
        self.tanh = nn.Tanh()

    def forward(self, x, intermediate_output=None):
        if intermediate_output is None:
            x = self.linear1(x)
        else:
            x = intermediate_output
        x = self.gelu1(x)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.gelu2(x)
        x = self.linear3(x)
        x = self.tanh(x)
        return x
```