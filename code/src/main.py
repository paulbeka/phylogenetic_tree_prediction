from get_tree_features import get_tree_features
from util.tree_manager import Tree, randomize_tree
from util.raxml_util import calculate_raxml
from networks.spr_likelihood_prediction_trainer import get_dataloader, train_value_network, test_value_network, test_model_ll_increase
from networks.gnn_network import load_tree, train_gnn_network

import random, dendropy, os
from tqdm import tqdm
import matplotlib.pyplot as plt


TRAIN_TEST_SPLIT = 0.8


def main():

	data_files = find_data_files("data/fast_tree_dataset")

	training_data = {
		"spr": [],
		"gnn": []
	}

	for i in range(int(len(training_data)*TRAIN_TEST_SPLIT)):
		tree = Tree(data_files[i]) 
		dataset, gnn_dataset, base_ll = create_dataset(tree, 2)
		training_data["spr"] += dataset
		training_data["gnn"] += gnn_dataset

	training_data["spr"] = get_dataloader(training_data["spr"])
	
	# For training this --> classify each of the nodes from most needed to move
	# to least needed. Then assign some sort of score to each
	# then train the GNN to identify which to move
	# because GNN needs to train several times on masked values
	spr_model = train_value_network(training_data["spr"])
	gnn_model = train_gnn_network(training_data["gnn"])

	# random_tree = randomize_tree(tree)
	# test_model_ll_increase(model, random_tree)


# Or maybe I should find the optimal branch, and only train with that
# it would be more accurate but the training time would be much longer
def create_dataset(tree, n_items):
	dataset = []
	gnn_dataset = []
	base_ll = []
	prev_ll = None
	for i in tqdm(range(n_items)):
		
		actionSpace = tree.find_action_space()
		action = random.choice(actionSpace)

		original_point = tree.perform_spr(action[0], action[1], return_parent=True)
		treeProperties = get_tree_features(tree, action[0], original_point)

		# rest = 0? --> ask supervisor
		gnn_data = load_tree(tree, original_point=original_point)
		gnn_dataset.append(gnn_data)

		# raxml_score = float(calculate_raxml(tree)["ll"])
		raxml_score = 0 # WINDOWS ONLY
		base_ll.append((i, raxml_score))

		if not prev_ll:
			score = abs(raxml_score)
		else:
			score = prev_ll - abs(raxml_score)

		dataset.append((treeProperties, score))

	return dataset, gnn_dataset, base_ll


def find_data_files(path):
	list_of_files = []
	for file in os.scandir(path):
		if file.name.split(".")[-1] == "fasta":
			list_of_files.append(f"{path}/{file.name.split('.')[0]}")
	return list_of_files

if __name__ == "__main__":
	main()