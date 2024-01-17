#######################################################################################
################################ GET THE TREE FEATURES ################################
#######################################################################################


# somehow get the pruned tree in here
def get_tree_features(tree, subtree, regrft_loc):
	main_tree_branch_tot = tree.tree.total_branch_length()
	subtree_branch_tot = subtree.total_branch_length()
	regrft_branch_tot = regrft_loc.total_branch_length()
	
	branch_dist = len(tree.tree.trace(subtree, regrft_loc))

	depths = tree.tree.depths(True)

	subtree_centrality = depths[subtree]
	regrft_centrality = depths[regrft_loc]

	features = {
		"main_tree_branch_tot": main_tree_branch_tot,
		"subtree_branch_tot": subtree_branch_tot,
		"regrft_branch_tot": regrft_branch_tot,
		"branch_dist": branch_dist,
		"subtree_centrality": subtree_centrality,
		"regrft_centrality": regrft_centrality
	}

	return features

