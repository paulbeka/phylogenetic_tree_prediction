from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor, Scorer
from Bio.Phylo import write, read
import dendropy
from dendropy.calculate import treecompare
from Bio import AlignIO
import csv


class Tree:

	def __init__(self):

		_, alignment = get_tree_and_alignment("data/fast_tree_dataset/COG527")

		with open("data/fast_tree_dataset/COG527.fasta", "w") as f:
			f.write(format(alignment, "fasta"))

		calculator = DistanceCalculator('identity')
		distMatrix = calculator.get_distance(alignment)
		treeConstructor = DistanceTreeConstructor()

		self.tree = treeConstructor.upgma(distMatrix)


	def find_action_space(self):
		nodes = []

		for clade in self.tree.find_clades():
			nodes.append(clade)

		actionSpace = []
		for node in nodes:
			for item in nodes:
				if node == item:
					break

				actionSpace.append((node, item))

		return actionSpace


	def perform_spr(self, subtree, regraft_location):
		parent = get_parent(self.tree, subtree)

		if parent is None:
			raise ValueError("Can't prune the root.")
	    
		parent.clades.remove(subtree)
	    
		if regraft_location:
			regraft_location.clades.append(subtree)
		else:
			self.tree.root.clades.append(subtree)
	    

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

def get_parent(tree, child_clade):
	node_path = tree.get_path(child_clade)
	try:
		return node_path[-2]
	except:
		return tree.root

def get_tree_and_alignment(loc):
	return (get_tree(loc), get_alignment(loc))