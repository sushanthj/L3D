import torch

# Create a tensor of shape (2, 3, 3)
obj_tensor = torch.arange(18).reshape(2, 3, 3)

print("obj_tensor:")
print(obj_tensor)

# Get the diagonal elements
diag_elements = torch.diagonal(obj_tensor, dim1=1, dim2=2)

print("\ndiag_elements:")
print(diag_elements)
