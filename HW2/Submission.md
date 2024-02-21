# Pix to Vox training

## Using Resnet18 and no batchnorm

- vox.pth
- ```python3
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


## Using ConvNext and Batchnorm

- vox_1.pth
- ```python3
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



# Image to Pointcloud

## Backbone Resnet18

<!-- insert image -->

## Backbone ConvNext with Batchnorm

<!-- insert_image -->

# 