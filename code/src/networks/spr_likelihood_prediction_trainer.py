import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm

from .spr_likelihood_network import SprScoreFinder


def train_test_split(dataset, batch_size=1):
	data = [(np.array(list(item[0].values())), item[1]) for item in dataset]
	train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
	# TODO: run multiple trees and keep some for test data
	test_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
	return train_loader, test_loader


def train_value_network(train_loader):

	num_epochs = 10
	batch_size = 1
	learning_rate = 0.0001

	model =  SprScoreFinder(batch_size).double()

	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	total_steps = len(train_loader)
	for epoch in range(num_epochs):
		print(f"Epoch: {epoch+1}/{num_epochs}")
		for i, (items, labels) in tqdm(enumerate(train_loader), desc="Training: ", total=len(train_loader)):

			outputs = model(items)
			loss = criterion(outputs.double(), labels.double())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print(f"Saving epoch {epoch+1}...")

	torch.save(model.state_dict(), "output")

	return model


def test_value_network(model, test_loader):

	with torch.no_grad():
		test_loss = 0
		for configs, labels in test_loader:
			outputs = model(configs.double())

			test_loss += criterion(outputs, labels)


		print(f"Total loss: {test_loss / len(test_loader)}")


def test_model_ll_increase(model, tree, n_iters=50):
	moves = []
	for i in range(n_iters):
		action_space = tree.find_action_space()
		best_move = None
		for action in action_space:
			output = model(action)
			if best_move == None:
				best_move = (action, output)
			elif output > best_move[1]:
				best_move = (action, output)

		tree.perform_spr(best_move[0][0], best_move[0][1])

	plt.plot(moves)
	plt.show()