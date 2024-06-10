import torch


nums = []
for i in range(10):
    nums.append(torch.rand(1))

# join the list of tensors into a single tensor
nums = torch.cat(nums)

print(nums)
