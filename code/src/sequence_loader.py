import dendropy
from dendropy.calculate import treecompare
from Bio import AlignIO
import csv


def get_alignment(loc):
	with open(f"{loc}.sim.p") as f:
		return AlignIO.read(f, 'phylip')


def get_alignment_sequence_dict(alignment):
	dictionary = {}
	for seq in alignment:
		dictionary[seq.name] = seq.seq
	return dictionary


def get_tree(loc):
	with open(f"{loc}.sim.trim.tree") as f:
		data = f.read()
		return dendropy.Tree.get(data=data, schema="newick")


def get_tree_and_alignment(loc):
	return (get_tree(loc), get_alignment(loc))


if __name__ == "__main__":
	get_tree_and_alignment("data/fast_tree_dataset/COG527")