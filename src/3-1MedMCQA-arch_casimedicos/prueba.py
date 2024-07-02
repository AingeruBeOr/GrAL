import torch

final = []

tensor = torch.tensor([1, 2, 3])
print(tensor.tolist())
final.extend(tensor.tolist())

tensor = torch.tensor([4, 5, 6])
print(tensor.tolist())
final.extend(tensor.tolist())

print(final)
