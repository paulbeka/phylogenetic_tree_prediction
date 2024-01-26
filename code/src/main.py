from get_tree_features import get_tree_features
from util.tree_manager import Tree, randomize_tree
from util.raxml_util import calculate_raxml
from networks.spr_likelihood_prediction_trainer import train_test_split, train_value_network, test_value_network, test_model_ll_increase
from networks.gnn_network import load_tree, train_gnn_network

import random, dendropy
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
	tree = Tree("data/fast_tree_dataset/COG527") 
	# dataset, base_ll = create_dataset(tree, 250)
	train_gnn_network(tree)


	
	# # For testing on Windows
	# dataset = [
	# 	({"main_tree_branch_tot": 0, "subtree_branch_tot": 0, "regrft_branch_tot": 0, "branch_dist": 0, "subtree_centrality": 0, "regrft_centrality": 0}, 0.0),
	# 	({"main_tree_branch_tot": 0, "subtree_branch_tot": 0, "regrft_branch_tot": 0, "branch_dist": 0, "subtree_centrality": 0, "regrft_centrality": 0}, 0.0),
	# 	({"main_tree_branch_tot": 0, "subtree_branch_tot": 0, "regrft_branch_tot": 0, "branch_dist": 0, "subtree_centrality": 0, "regrft_centrality": 0}, 0.0),
	# ]

	# # show that the ll is going down as we random search
	# plt.plot([x[1] for x in base_ll])
	# plt.show()

	# train, test = train_test_split(dataset)
	# model = train_value_network(train)
	# test_value_network(model, test)

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
		gnn_dataset.append((original_point, 1))

		raxml_score = float(calculate_raxml(tree)["ll"])
		base_ll.append((i, raxmlscore))

		if not prev_ll:
			score = abs(raxml_score)
		else:
			score = prev_ll - abs(raxml_score)

		dataset.append((treeProperties, score))

	return dataset, base_ll


if __name__ == "__main__":
	main()