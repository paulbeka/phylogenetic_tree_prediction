from sequence_loader import get_tree_and_alignment, get_alignment_sequence_dict
import torch
import torch.nn as nn
import dendropy
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor, Scorer
from dendropy.calculate import treecompare#
from Bio.Phylo import write, read
import copy
import random
from networks import neural_network
from tqdm import tqdm


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
	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# torch.cuda.set_device(device)
	# model.to(device)
	# model.cuda(device)

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
		for configs, labels in test_loader:
			ouputs = model(configs)

			# _, predictions = torch.max(outputs, 1)
			test_loss += criterion(outputs, labels)

		print(f"Total loss: {test_loss/len(test_loader)}")


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
	treeProps = getTreeProperties(tree)
	originalTreeProps = getTreeProperties(originalTree)

	length_diff = abs(treeProps["total_branch_length"] - originalTreeProps["total_branch_length"])
	n_nodes_diff = abs(treeProps["n_nodes"] - originalTreeProps["n_nodes"])
	avg_terminal_distance_diff = abs(treeProps["avg_terminal_distance"] - originalTreeProps["avg_terminal_distance"])

	return (0.4 * length_diff) + (0.3 * n_nodes_diff) + (0.3 * avg_terminal_distance_diff)


def getTreeProperties(tree):
	properties = {}
	properties['total_branch_length'] = tree.total_branch_length()
	
	n_terminals = 0
	total_distance = 0
	for terminal in tree.get_terminals():
		total_distance += tree.distance(terminal)
		n_terminals += 1
	
	properties['n_nodes'] = len(tree.get_nonterminals()) + n_terminals
	properties['avg_terminal_distance'] = total_distance / n_terminals

	return properties


# TODO: check that the generation is not just re-generating the same tree again and again. !!!
def generateDataset(originalTree, n_items):
	dataset = []
	currentTree = copy.copy(originalTree)
	for i in tqdm(range(n_items)):
		possible_actions = find_action_space(currentTree)
		action = random.choice(possible_actions)
		currentTree = perform_spr(currentTree, action[0], action[1])
		inputMatrix = list(getTreeProperties(currentTree).values())
		dataset.append((torch.FloatTensor(inputMatrix), torch.FloatTensor([getTreeScore(currentTree, originalTree)])))

	return dataset


def displayMatplotlib(dataset):
	pass


# UTILITY - GET PARENT OF A CLADE
def get_parent(tree, child_clade):
	node_path = tree.get_path(child_clade)
	try:
		return node_path[-2]
	except:
		return tree.root


if __name__ == "__main__":
	main()