import torch
from networks.gnn_network import load_tree
from get_tree_features import get_tree_features


N_TOP = 3


def train_algorithm(tree, n_iters):
	pass


def run_algorithm(tree, spr_model, gnn_model, n_iters):
	ll_path = []
	for i in range(n_iters):
		tree_gnn_data = load_tree(tree)
		_, top = torch.topk(gnn_model(tree_gnn_data.x, tree_gnn_data.edge_index)[:, 0], N_TOP)

		candidates = []
		for item in top:
			node = next(iter(tree.tree.find_clades(name=tree_gnn_data.node[item.item()])))  # get the clade object
			candidates += tree.find_action_space(rootNode=node)

		best_move = None
		for candidate in candidates:
			features = get_tree_features(tree, candidate[0], candidate[1])
			feature_array = torch.Tensor(list(features.values()))
			score = spr_model(feature_array)
			if best_move and best_move[1] < score:
				best_move = (candidate, score)
			else:
				best_move = (candidate, score)

		tree.perform_spr(best_move[0][0], best_move[0][1])
		ll_path.append(best_move[1].item())

	print(ll_path)
	return tree


def test_algorithm(starting_tree):
	pass


def load_saved_models():
	pass

