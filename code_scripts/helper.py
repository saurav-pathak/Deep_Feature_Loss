import torch

def l1_Loss(current, target):
	return torch.mean(torch.abs(current - target))

def l2_Loss(current, target):
	return torch.mean(torch.abs(current - target)**2)