import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm

from .spr_likelihood_network import SprScoreFinder


def train_value_network(dataset):

	num_epochs = 10
	batch_size = 1
	learning_rate = 0.0001

	model =  SprScoreFinder(batch_size).double()

	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	# Load data into torch format
	data = [(np.array(list(item[0].values())), item[1]) for item in dataset]
	train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

	total_steps = len(train_loader)
	for epoch in range(num_epochs):
		print(f"Epoch: {epoch+1}/{num_epochs}")
		for i, (items, labels) in tqdm(enumerate(train_loader), desc="Training: ", total=len(train_loader)):

			outputs = model(items.double())
			loss = criterion(outputs.double(), labels.double())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print(f"Saving epoch {epoch+1}...")

		torch.save(model.state_dict(), f"deconstructScoreOutputFile_{epoch+1}")

	return model