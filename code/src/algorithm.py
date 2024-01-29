

def train_algorithm(tree, n_iters):
	pass


def apply_algorithm(tree, n_iters):
	for i in range(n_iters):
		tree_gnn_data = load_tree(tree)
		gnn_model(tree_gnn_data)
		# get the maximum value
		top_candidates = []

		# get actions with top candidates
		# ...

		best_move = None
		for candidate in top_candidates:
			score = spr_model(candidate)
			if best_move and best_move[1] < score:
				best_move = (candidate, score)
			else:
				best_move = (candidate, score)

		tree.perform_spr(action[0][0], action[0][1])

	return tree


def load_saved_models():
	pass

