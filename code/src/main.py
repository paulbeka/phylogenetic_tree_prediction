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

	actionSpace = find_action_space(tree)

	# subtree_to_prune = next(iter(tree.find_clades(name="N177")))
	# regraft_location = next(iter(tree.find_clades(name="N155")))
	# new_tree = perform_spr(tree, subtree_to_prune, regraft_location)

	print(getTreeScore(tree, original_tree))



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
	calculator = DistanceCalculator('identity')
	dm1 = calculator.get_distance(tree)
	dm2 = calculator.get_distance(originalTree)

	matrix1 = dm1.matrix_form()
	matrix2 = dm2.matrix_form()
	print(matrix1, matrix2)
	print(dir(tree))
	return tree.robinson_foulds(originalTree)[0]


# UTILITY - GET PARENT OF A CLADE
def get_parent(tree, child_clade):
    node_path = tree.get_path(child_clade)
    return node_path[-2]


if __name__ == "__main__":
	main()