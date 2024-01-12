from sequence_loader import get_tree_and_alignment, get_alignment_sequence_dict
import torch
import torch.nn as nn
import dendropy
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor, Scorer
from dendropy.calculate import treecompare
from Bio.Phylo import write, read
from Bio import Phylo
import copy
import random
from networks import neural_network
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from dendropy.interop import raxml
import networkx as nx
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


actions = []


def main():

	_, alignment = get_tree_and_alignment("data/fast_tree_dataset/COG527")
	
	calculator = DistanceCalculator('identity')
	distMatrix = calculator.get_distance(alignment)
	treeConstructor = DistanceTreeConstructor()

	original_tree = read("data/fast_tree_dataset/COG527.sim.trim.tree", "newick")
	tree = treeConstructor.upgma(distMatrix)


	# LOAD THE DATA
	n_train, n_test = 1000, 200

	print("Loading training dataset...")
	train_dataset = generateDataset(original_tree, n_train)
	train_loader = torch.utils.data.DataLoader(
		dataset=train_dataset, 
		batch_size=1, 
		shuffle=True)

	displayMatplotlib(train_dataset)

	print("Loading test dataset...")
	test_dataset = generateDataset(original_tree, n_test)
	test_loader = torch.utils.data.DataLoader(
		dataset=test_dataset,
		batch_size=1, 
		shuffle=True
	)
	
	# TRAIN THE NEURAL NETWORK

	num_epochs = 10
	batch_size = 1
	learning_rate = 0.0001

	model =  neural_network.ScoreFinder(batch_size)

	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	total_steps = len(train_loader)
	for epoch in range(num_epochs):
		print(f"Epoch: {epoch+1}/{num_epochs}")
		for i, (configs, labels) in tqdm(enumerate(train_loader), desc="Training: ", total=len(train_loader)):
			outputs = model(configs)
			loss = criterion(outputs, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()


	with torch.no_grad():
		correct, samples = 0, 0
		test_loss = 0
		out = []
		for configs, labels in test_loader:
			ouputs = model(configs)
			out.append(outputs)

			# _, predictions = torch.max(outputs, 1)
			test_loss += criterion(outputs, labels)

		print(f"Total loss: {test_loss/len(test_loader)}")

		print(out)


def find_action_space(tree):
	nodes = []

	for clade in tree.find_clades():
		nodes.append(clade)

	actionSpace = []
	for node in nodes:
		for item in nodes:
			if node == item:
				break

			actionSpace.append((node, item))

	return actionSpace


def perform_spr(tree, subtree, regraft_location):
	parent = get_parent(tree, subtree)

	if parent is None:
		raise ValueError("Can't prune the root.")
    
	parent.clades.remove(subtree)
    
	if regraft_location:
		regraft_location.clades.append(subtree)
	else:
		tree.root.clades.append(subtree)
    
	return tree


def getTreeScore(tree, originalTree):
	# likelihood score ? BioPython library might have it
	treeProps = getTreeProperties(tree)
	originalTreeProps = getTreeProperties(originalTree)

	length_diff = abs(treeProps["total_branch_length"] - originalTreeProps["total_branch_length"])
	max_branch_length_diff = abs(treeProps["max_branch_length"] - originalTreeProps["max_branch_length"])
	avg_terminal_distance_diff = abs(treeProps["avg_terminal_distance"] - originalTreeProps["avg_terminal_distance"])
	
	return (avg_terminal_distance_diff * 10) + (max_branch_length_diff * 0.1)


def getTreeProperties(tree):
	properties = {}
	properties['total_branch_length'] = tree.total_branch_length()

	max_length = 0
	n_terminals = 0
	total_distance = 0
	for element in tree.find_elements():
		if element.is_terminal():
			total_distance += tree.distance(element)
			n_terminals += 1
		if hasattr(element, "branch_length"):
			if element.branch_length > max_length:
				max_length = element.branch_length
	
	properties['max_branch_length'] = max_length
	properties['avg_terminal_distance'] = total_distance / n_terminals

	return properties


# TODO: check that the generation is not just re-generating the same tree again and again. !!!
def generateDataset(originalTree, n_items):
	dataset = []
	currentTree = copy.deepcopy(originalTree)
	for i in tqdm(range(n_items)):
		possible_actions = find_action_space(currentTree)
		action = random.choice(possible_actions)
		currentTree = copy.deepcopy(perform_spr(currentTree, action[0], action[1]))
		inputMatrix = list(getTreeProperties(currentTree).values())
		dataset.append((torch.FloatTensor(inputMatrix), torch.FloatTensor([getTreeScore(currentTree, originalTree)])))

	return dataset


def displayMatplotlib(dataset):
	dat = [x[1] for x in dataset]
	x_axis = list(range(len(dataset)))
	
	plt.plot(x_axis, dat)
	plt.show()


# UTILITY - GET PARENT OF A CLADE
def get_parent(tree, child_clade):
	node_path = tree.get_path(child_clade)
	try:
		return node_path[-2]
	except:
		return tree.root


if __name__ == "__main__":
	# main()

	_, alignment = get_tree_and_alignment("data/fast_tree_dataset/COG527")
	
	calculator = DistanceCalculator('identity')
	distMatrix = calculator.get_distance(alignment)
	treeConstructor = DistanceTreeConstructor()

	original_tree = read("data/fast_tree_dataset/COG527.sim.trim.tree", "newick")
	tree = treeConstructor.upgma(distMatrix)

	# This is RAxML for DNA
	data = dendropy.DnaCharacterMatrix.get(
	    path="pythonidae.nex",
	    schema="nexus")
	rx = raxml.RaxmlRunner()
	tree = rx.estimate_tree(
	        char_matrix=data,
	        raxml_args=["--no-bfgs"])
	print(tree.as_string(schema="newick"))


	graph = Phylo.to_networkx(tree)

	data = from_networkx(graph)

	class GNN(torch.nn.Module):
	    def __init__(self, input_dim, hidden_dim, output_dim):
	        super(GNN, self).__init__()
	        self.conv1 = GCNConv(input_dim, hidden_dim)
	        self.conv2 = GCNConv(hidden_dim, output_dim)

	    def forward(self, data):
	    	# Here, extract a meaninful feature to do something with it
	        x, edge_index = data.weight, data.edge_index
	        x = F.relu(self.conv1(x, edge_index))
	        x = F.relu(self.conv2(x, edge_index))
	        return F.log_softmax(x, dim=1)

	input_dim = 1
	hidden_dim = 64
	output_dim = 2

	model = GNN(input_dim, hidden_dim, output_dim)

	output = model(data)
	print(output)