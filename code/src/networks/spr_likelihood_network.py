import torch
import torch.nn as nn
import torch.nn.functional as F


class SprScoreFinder(nn.Module):
	def __init__(self, batch_size):
		super(SprScoreFinder, self).__init__()
		
		self.firstLayer = nn.Linear(6, 10)
		self.secondLayer = nn.Linear(10, 10)
		self.finalLayer = nn.Linear(10, 1)

		self.relu = nn.ReLU()


	def forward(self, x):
		x = self.relu(self.firstLayer(x))
		x = self.relu(self.secondLayer(x))
		return self.finalLayer(x)