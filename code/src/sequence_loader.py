import dendropy
from dendropy.calculate import treecompare
from Bio import AlignIO
import csv


def get_tree_and_alignment():

	alignment = None
	tree = None

	with open("data/fast_tree_dataset/COG527.sim.p") as f:
		alignment = AlignIO.read(f, 'phylip')

	with open("data/fast_tree_dataset/COG527.sim.trim.tree") as f:
		data = f.read()
		tree = dendropy.Tree.get(data=data, schema="newick")

	return (tree, alignment)


if __name__ == "__main__":
	get_tree_and_alignment()