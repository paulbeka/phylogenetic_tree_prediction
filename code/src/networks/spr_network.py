import torch, random, copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("..") 

from util.raxml_util import calculate_raxml
from get_tree_features import get_tree_features


class SprScoreFinder(nn.Module):
	def __init__(self, batch_size):
		super(SprScoreFinder, self).__init__()
		
		self.firstLayer = nn.Linear(6, 20)
		self.secondLayer = nn.Linear(20, 20)
		self.thirdLayer = nn.Linear(20, 10)
		self.finalLayer = nn.Linear(10, 1)

		self.silu = nn.SiLU()

	def forward(self, x):
		x = self.silu(self.firstLayer(x))
		x = self.silu(self.secondLayer(x))
		x = self.silu(self.thirdLayer(x))
		x = self.finalLayer(x)
		return x


def train_value_network(train_loader, test=None, generated_spr=None,
	n_epochs=75, batch_size=1, lr=0.0001):

	model =  SprScoreFinder(batch_size).double()

	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	best_model = (None, 9999999)

	total_steps = len(train_loader)
	for epoch in range(n_epochs):
		print(f"Epoch: {epoch+1}/{n_epochs}")
		for i, (items, labels) in tqdm(enumerate(train_loader), desc="Training: ", total=len(train_loader)):

			outputs = model(items)
			loss = criterion(outputs.double(), labels)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if test:
			loss = test_top_with_raxml(model, test, generated_spr)
			if loss > best_model[1]:
				best_model = (copy.deepcopy(model.state_dict()), loss)

	return best_model


def test_value_network(model, test_loader):
	criterion = nn.MSELoss()
	with torch.no_grad():
		test_loss = 0
		for configs, labels in test_loader:
			for x in item:
				action = x[0]
				features = get_tree_features(tree, action[0], action[1])
				feature_array = torch.Tensor(list(features.values())).double()
				score = model(feature_array).item()

				test_loss += criterion(x[1], score)

		print(f"Total loss: {test_loss / len(test_loader)}")
	
	return test_loss / len(test_loader)	


# Look at the top move from the set of moves which was done during random walk
# Avoids having to re-calculate raxml-ng scores for each tree (less compute time)
# NOTE: IT ACTUALLY LOOKS AT TOP 5, NOT TOP 10!!!
def test_top_10(model, test_dataset):
	n_top_10 = 0

	with torch.no_grad():
		for group in test_dataset:
			group = [(x[0], x[1], i) for i, x in enumerate(group)]
			group.sort(key=lambda x: x[1])
			preds = []
			max_pred = None
			for item in group:
				x = model(item[0]).item()
				if max_pred == None or max_pred[1] < x:
					max_pred = (item[0], x, item[2])

			if max_pred[2] in [item[2] for item in group[-5:]]:
				n_top_10 += 1

	return (n_top_10 / (len(test_dataset)*len(test_dataset[0])))*100


# warning: slow
# input randomized trees for the test dataset
def test_top_with_raxml(model, test_dataset, generated_raxml_vals):
	n_top = 10
	average = []
	with torch.no_grad():
		for i, tree in enumerate(test_dataset):
			n_found = 0
			actionSpace = tree.find_action_space()
			model_ranking = []
			for action in actionSpace:
				# no need for deepcopy on features
				features = get_tree_features(tree, action[0], action[1])
				feature_array = torch.Tensor(list(features.values())).double()
				score = model(feature_array).item()
				model_ranking.append((action, score))

			model_ranking.sort(key=lambda x: x[1])
			
			top_true = [x[0] for x in generated_raxml_vals[i][-n_top:]]
			top_model = [x[0] for x in model_ranking[-n_top:]]

			for item in top_model:
				if item in top_true:
					n_found += 1

			average.append(n_found / n_top)

		acc = (sum(average) / len(average))*100
		print(f"SPR score found: {acc:.2f}%")
		return acc


def generate_top_raxml_test_dataset(test_dataset):
	true_ranking = []
	for tree in test_dataset:
		actionSpace = tree.find_action_space()
		true_ranking_temp = []
		for action in actionSpace:
			treeCopy = copy.deepcopy(tree)
			actionCopy = copy.deepcopy(action)
			treeCopy.perform_spr(actionCopy[0], actionCopy[1], deepcopy=True)
			try:
				raxml_score = float(calculate_raxml(treeCopy)["ll"])
				# raxml_score = 0
				true_ranking_temp.append((action, raxml_score))
			except:
				raise Exception("Use on computer with raxml-ng!")
		
		true_ranking_temp.sort(key=lambda x: x[1])
		true_ranking.append(true_ranking_temp)

	return true_ranking


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
	plt.plot(sorted(true_vals), label="True")
	plt.plot(pred_vals, label="Predicted")
	plt.legend()
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


def optimize_spr_network(train, test):
	epoch_values = [10, 20]
	lr_values = [0.0001, 0.001, 0.0005]
	batch_values = [1, 5, 20]

	combinations_list = []
	for epoch in epoch_values:
		for lr in lr_values:
			for batch_size in batch_values:
				print(f"Training with: Epochs: {epoch}, LR: {lr}, Batch Size: {batch_size}")
				model, acc = train_value_network(train, test=test,
					n_epochs=epoch, batch_size=batch_size, lr=lr)
				print(f"Accuracy found: {acc}")

				combinations_list.append((batch_size, lr, epoch, model, acc))

	combinations_list.sort(key=lambda x: x[-1])
	best = combinations_list[-1]
	print(f"The best model found had: batch size: {best[0]}, LR: {best[1]}, epochs: {epoch}")
	return best[-2]


##### UTILITY FUNCTIONS #####
def get_dataloader(dataset, batch_size=1):
	data = [(np.array(list(item[0].values())), item[1]) for item in dataset]
	train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
	return train_loader
