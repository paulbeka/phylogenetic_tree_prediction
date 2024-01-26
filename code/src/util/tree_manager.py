from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor, Scorer
from Bio.Phylo import write, read
import dendropy
from dendropy.calculate import treecompare
from Bio import AlignIO
from Bio.Phylo.BaseTree import Clade
import csv, random


class Tree:

	# TODO: THIS IS NOT LOADING THE ORIGINAL TREE, BUT INSTEAD CALCULATING THE UPGMA TREE
	def __init__(self, loc):

		tree, alignment = get_tree_and_alignment(loc)

		filename = loc.split(".")[0]
		with open(f"{filename}.fasta", "w") as f:
			f.write(format(alignment, "fasta"))

		calculator = DistanceCalculator('identity')
		distMatrix = calculator.get_distance(alignment)
		treeConstructor = DistanceTreeConstructor()

		self.alignment = get_alignment_sequence_dict(alignment)
		self.tree = treeConstructor.upgma(distMatrix)
		# self.tree = tree

	def find_action_space(self):
		# TODO: !!! for small trees, this needs to be fixed
		nodes = [node for node in self.tree.find_clades() if (node != self.tree.root) and (node not in self.tree.root.clades)]

		actionSpace = []
		for node in nodes:
			for item in nodes:
				if node == item:
					break

				if (node in item.clades) or (item in node.clades):
					break

				actionSpace.append((node, item))

		return actionSpace


	# TODO: Fix operation close to the root of the tree
	def perform_spr(self, subtree, regraft_location, return_parent=False):
		parent = get_parent(self.tree, subtree)

		if parent is None:
			raise ValueError("Can't prune the root.")

		# remove the subtree
		parent.clades.remove(subtree)

		# restructure tree to remove unifurcations
		child = parent.clades[0]
		grandpa = get_parent(self.tree, parent)
		grandpa.clades.remove(parent)
		grandpa.clades.append(child)

		# graft new clade
		new_clade = Clade(branch_length=0.1, name=parent.name)

		regraft_parent = get_parent(self.tree, regraft_location)
		regraft_parent.clades.remove(regraft_location)
		new_clade.clades.append(regraft_location)
		new_clade.clades.append(subtree)
		regraft_parent.clades.append(new_clade)

		if return_parent:
			return child


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


def randomize_tree(tree):
	for i in range(random.randint(10, 50)):
		next_action = random.choice(tree.find_action_space())
		tree.perform_spr(next_action[0], next_action[1])
	return tree