from sequence_loader import get_tree_and_alignment, get_alignment_sequence_dict
import torch
import dendropy
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor, Scorer
from dendropy.calculate import treecompare#
from Bio.Phylo import write, read
import copy


actions = []


def main():
	
	_, alignment = get_tree_and_alignment("data/fast_tree_dataset/COG527")
	
	calculator = DistanceCalculator('identity')
	distMatrix = calculator.get_distance(alignment)
	treeConstructor = DistanceTreeConstructor()

	original_tree = read("data/fast_tree_dataset/COG527.sim.trim.tree", "newick")
	tree = treeConstructor.upgma(distMatrix)

	scorer = Scorer()

	for i in range(1):
		actionSpace = find_action_space(tree)
		getTreeScore(tree, original_tree)


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


# UTILITY - GET PARENT OF A CLADE
def get_parent(tree, child_clade):
    node_path = tree.get_path(child_clade)
    return node_path[-2]


if __name__ == "__main__":
	main()