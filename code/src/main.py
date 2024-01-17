from get_tree_features import get_tree_features
from util.tree_manager import Tree
from util.raxml_util import calculate_raxml
from networks.spr_likelihood_prediction_trainer import train_value_network

import random, dendropy
from tqdm import tqdm


def main():
	tree = Tree() # add str input for loc
	dataset = create_dataset(tree, 100)
	
	# For testing on Windows
	# dataset = [
	# 	({"main_tree_branch_tot": 0, "subtree_branch_tot": 0, "regrft_branch_tot": 0, "branch_dist": 0, "subtree_centrality": 0, "regrft_centrality": 0}, 0),
	# 	({"main_tree_branch_tot": 0, "subtree_branch_tot": 0, "regrft_branch_tot": 0, "branch_dist": 0, "subtree_centrality": 0, "regrft_centrality": 0}, 0),
	# 	({"main_tree_branch_tot": 0, "subtree_branch_tot": 0, "regrft_branch_tot": 0, "branch_dist": 0, "subtree_centrality": 0, "regrft_centrality": 0}, 0),
	# ]

	model = train_value_network(dataset)
	test_value_network(model)


# Or maybe I should find the optimal branch, and only train with that
# it would be more accurate but the training time would be much longer
def create_dataset(tree, n_items):
	dataset = []
	for i in tqdm(range(n_items)):
		actionSpace = tree.find_action_space()
		action = random.choice(actionSpace)
		treeProperties = get_tree_features(tree, action[0], action[1])
		# TODO: fix SPR fringe errors
		tree.perform_spr(action[0], action[1])
		score = float(calculate_raxml(tree)["ll"])
		dataset.append((treeProperties, score))

	return dataset



if __name__ == "__main__":
	main()