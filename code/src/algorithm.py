import torch
from networks.gnn_network import load_tree
from get_tree_features import get_tree_features
from util.raxml_util import calculate_raxml
import matplotlib.pyplot as plt


N_TOP = 3


def train_algorithm(tree, n_iters):
	pass


# This is a greedy algorithm - is this good?	
def run_algorithm(tree, spr_model, gnn_model, n_iters, find_true_ll_path=False):
	ll_path = []
	true_ll_path = []
	max_ll = -999999999
	for i in range(n_iters):
		
		loop = 0
		candidates = []
		while candidates == []:
			candidates = find_candidates(tree, gnn_model, N_TOP+loop)
			loop += 1

		best_move = None
		for candidate in candidates:
			features = get_tree_features(tree, candidate[0], candidate[1])
			feature_array = torch.Tensor(list(features.values()))
			score = spr_model(feature_array).item()
			if best_move and best_move[1] < score:
				best_move = (candidate, score)
			else:
				best_move = (candidate, score)

		tree.perform_spr(best_move[0][0], best_move[0][1])
		ll_path.append(best_move[1])

		if find_true_ll_path:
			score = float(calculate_raxml(tree)["ll"])
			max_ll = max(max_ll, score)
			true_ll_path.append(max_ll)

	if find_true_ll_path:
		return tree, true_ll_path
	else:
		return tree, [0]*n_iters


def find_candidates(tree, gnn_model, N_TOP):
	tree_gnn_data = load_tree(tree)
	_, top = torch.topk(gnn_model(tree_gnn_data.x, tree_gnn_data.edge_index)[:, 0], N_TOP)

	candidates = []
	for item in top:
		node = next(iter(tree.tree.find_clades(name=tree_gnn_data.node[item.item()])))  # get the clade object
		if node == tree.tree.root or node in tree.tree.root.clades:
			continue
		candidates += tree.find_action_space(rootNode=node)

	return candidates


def find_candidates_with_node_network(tree, node_model, N_TOP):
	data = load_node_data(tree, generate_true_ratio=True)
	scored = [node_model(x[0]).item() for x in data]
	# scored.


def test_algorithm(starting_tree, original_score, spr_model, gnn_model):
	max_n_iters = 50
	tree, path = run_algorithm(starting_tree, spr_model, gnn_model, max_n_iters, find_true_ll_path=False)
	
	plt.plot([original_score]*max_n_iters)
	plt.plot(path)
	plt.title("Likelihood over iterations")
	plt.xlabel("Number of iterations")
	plt.ylabel("Likelihood")
	plt.savefig("alg_output")

	return tree


# def train_algorithm_reinforcement_learning(data, spr_model, gnn_model, n_iters):
# 	path = []
# 	for i in range(n_iters):
# 		for tree in dataset:
# 			true_tree = None # set this to the real tree
# 			final_tree = run_algorithm(tree, spr_model, gnn_model, n_iters, find_true_ll_path=False)
# 			# get ll score of the final tree
# 			if final_tree == true_tree:
# 				# then give a good score to the network
# 				score = 100
# 				# How do I assign it to be DRL?




def load_saved_models():
	pass

