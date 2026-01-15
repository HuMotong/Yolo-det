import torch

print("PyTorch:", torch.__version__)
print("CUDA in torch:", torch.version.cuda)
print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0))
