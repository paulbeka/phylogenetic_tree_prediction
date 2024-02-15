import torch, random, copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("..") 

from .spr_likelihood_network import SprScoreFinder
from util.raxml_util import calculate_raxml
from get_tree_features import get_tree_features


def train_value_network(train_loader, test=None):

	num_epochs = 10
	batch_size = 1
	learning_rate = 0.0001

	model =  SprScoreFinder(batch_size).double()

	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	best_model = (None, 9999999)

	total_steps = len(train_loader)
	for epoch in range(num_epochs):
		print(f"Epoch: {epoch+1}/{num_epochs}")
		for i, (items, labels) in tqdm(enumerate(train_loader), desc="Training: ", total=len(train_loader)):

			outputs = model(items.double())
			loss = criterion(outputs.double(), labels.unsqueeze(1).double())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if test:
			loss = test_value_network(model, test)
			if loss < best_model[1]:
				best_model = (model, loss)

		print(f"Saving epoch {epoch+1}...")

	return model


def test_value_network(model, test_loader):
	criterion = nn.MSELoss()
	with torch.no_grad():
		test_loss = 0
		for configs, labels in test_loader:
			outputs = model(configs.double())

			test_loss += criterion(outputs, labels.unsqueeze(1))

		print(f"Total loss: {test_loss / len(test_loader)}")
	
	return test_loss / len(test_loader)	


def compare_score(model, test_loader):
	true_vals = []
	pred_vals = []
	with torch.no_grad():
		test_loss = 0
		for configs, labels in test_loader:
			outputs = model(configs.double())
			true_vals.append(labels[0].item())
			pred_vals.append(outputs[0][0].item())
 
 	pred_vals = [x for _, x in sorted(zip(true_vals, pred_vals))]
	plt.plot(sorted(true_vals))
	plt.plot(pred_vals)
	plt.show()


def test_model_ll_increase(model, tree, n_iters=50):
	moves = []
	n_correct = 0
	for i in tqdm(range(n_iters)):
		action_space = random.sample(tree.find_action_space(), 10)
		best_move = None
		ground_truth = None
		for action in action_space:
			input_tensor = torch.tensor(list(get_tree_features(tree, action[0], action[1]).values()))
			output = model(input_tensor.double())
			if best_move == None:
				best_move = (action, output)
			elif output > best_move[1]:
				best_move = (action, output)

			# Ground truth calculation
			treeCopy = copy.deepcopy(tree)
			subtreeCopy = next(iter(treeCopy.tree.find_clades(target=action[0].name)))
			regraftCopy = next(iter(treeCopy.tree.find_clades(target=action[1].name)))
			treeCopy.perform_spr(subtreeCopy, regraftCopy)
			raxml_score = float(calculate_raxml(treeCopy)["ll"])
			if ground_truth and ground_truth[1] < raxml_score:
				ground_truth = (action, raxml_score)
			else:
				ground_truth = (action, raxml_score)

		moves.append(best_move[1].item())
		if best_move[0] == ground_truth[0]:
			n_correct += 1
		tree.perform_spr(best_move[0][0], best_move[0][1])

	print(f"Out of {str(n_iters)} moves, the best move was selected {str(n_correct)} times.")
	print(f"Accuracy: {((n_correct/n_iters) * 100):.2f}%")

	plt.plot(moves)
	plt.show()


def get_dataloader(dataset, batch_size=1):
	data = [(np.array(list(item[0].values())), item[1]) for item in dataset]
	train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
	return train_loader


def train_test_split(dataset, batch_size=1):
	data = [(np.array(list(item[0].values())), item[1]) for item in dataset]
	train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
	# TODO: run multiple trees and keep some for test data
	test_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
	return train_loader, test_loader
	