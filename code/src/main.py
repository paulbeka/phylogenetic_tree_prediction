from sequence_loader import get_tree_and_alignment, get_alignment_sequence_dict
import torch
import dendropy
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor, Scorer
from dendropy.calculate import treecompare#
from Bio.Phylo import write, read


actions = []


def main():
	
	tree1, alignment = get_tree_and_alignment("data/fast_tree_dataset/COG527")
	
	calculator = DistanceCalculator('identity')
	distMatrix = calculator.get_distance(alignment)
	treeConstructor = DistanceTreeConstructor()

	tree1 = read("data/fast_tree_dataset/COG527.sim.trim.tree", "newick")
	tree2 = treeConstructor.upgma(distMatrix)

	scorer = Scorer()
	# print(scorer.get_score(tree1, tree2))

	traverse_tree(tree2.root)
	find_action_space(tree2)


def find_action_space(tree):
	nodes = []
	for action in actions:
		if str(action)[0] == "N":
			nodes.append(action)

	print(nodes)
	common_ancestor = tree.common_ancestor(nodes[0], nodes[1])
	child1, child2 = common_ancestor.clades
	common_ancestor.clades = [child2, child1]


def traverse_tree(clade, depth=0):
	global actions
	for child in clade.clades:
		actions.append(child)
		traverse_tree(child, depth + 1)



if __name__ == "__main__":
	main()