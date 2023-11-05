import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
	def __init__(self, batch_size):
		super(NeuralNetwork, self).__init__()
		
		self.firstLayer = F.Linear()
		self.secondLayer = F.Linear()
		self.finalLayer = F.Linear()

		self.relu = F.relu()


	def forward(self, x):
		pass