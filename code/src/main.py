from sequence_loader import get_tree_and_alignment, get_alignment_sequence_dict
import torch
import dendropy
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor, Scorer
from dendropy.calculate import treecompare


def main():
	
	tree1, alignment = get_tree_and_alignment("data/fast_tree_dataset/COG527")
	
	calculator = DistanceCalculator('identity')
	distMatrix = calculator.get_distance(alignment)

	treeConstructor = DistanceTreeConstructor()
	upgma_tree = treeConstructor.upgma(distMatrix)

	# classBio.Phylo.TreeConstruction.Scorer
	# maybe this function can be used to score the difference?
	# no, we need to turn the tree into a dendropy tree so we can then score it using their fancy techniques.

	scorer = Scorer()
	print(dir(scorer))
	print(scorer.get_score(upgma_tree, alignment))


if __name__ == "__main__":
	main()