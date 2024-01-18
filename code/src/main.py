from get_tree_features import get_tree_features
from util.tree_manager import Tree
from util.raxml_util import calculate_raxml
from networks.spr_likelihood_prediction_trainer import train_value_network

import random, dendropy
from tqdm import tqdm


def main():
	tree = Tree() # add str input for loc
	dataset, base_ll = create_dataset(tree, 100)
	
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
	base_ll = []
	prev_ll = None
	for i in tqdm(range(n_items)):
		actionSpace = tree.find_action_space()
		action = random.choice(actionSpace)

		original_point = tree.perform_spr(action[0], action[1], return_parent=True)
		treeProperties = get_tree_features(tree, action[0], original_point)

		base_ll.append((i, float(ll)))

		if not prev_ll:
			score = float(abs(calculate_raxml(tree)["ll"]))
		else:
			score = prev_ll - abs(float(calculate_raxml(tree)["ll"]))

		dataset.append((treeProperties, score))

	return dataset, basell



if __name__ == "__main__":
	main()