from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor, Scorer
from Bio.Phylo import write, read
import dendropy
from dendropy.calculate import treecompare
from Bio import AlignIO
from Bio.Phylo.BaseTree import Clade
import csv, random


class Tree:

	def __init__(self, loc):

		self.location = loc
		tree, alignment = get_tree_and_alignment(loc)

		filename = loc.split(".")[0]
		with open(f"{filename}.fasta", "w") as f:
			f.write(format(alignment, "fasta"))

		self.alignment = get_alignment_sequence_dict(alignment)
		self.tree = read(f"{loc}.fasta.raxml.bestTree", format="newick")

		if len(self.tree.root.clades) > 2:
			new_clade = Clade(branch_length=0.000001, name="bigdaddy")
			new_clade.clades = self.tree.root.clades[:2]
			for clade in new_clade.clades:
				self.tree.root.clades.remove(clade)
			self.tree.root.clades.append(new_clade)

		count = 0
		for node in [self.tree.root, *self.tree.find_elements()]:
			if node.name == None:
				setattr(node, "name", str(count))
				count += 1
		self.n_nodes = count


	def find_action_space(self):
		# TODO: !!! for small trees, this needs to be fixed
		nodes = [node for node in self.tree.find_clades() if (node != self.tree.root) and (node not in self.tree.root.clades)]

		count = 0

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
		subtree = list(self.tree.find_elements(name=subtree.name))[0]
		regraft_location = list(self.tree.find_elements(name=regraft_location.name))[0]

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
		new_clade = Clade(branch_length=0.1, name=parent.name)	# constant branch length of 0.1 

		regraft_parent = get_parent(self.tree, regraft_location)
		regraft_parent.clades.remove(regraft_location)
		new_clade.clades.append(regraft_location)
		new_clade.clades.append(subtree)
		regraft_parent.clades.append(new_clade)

		if return_parent:
			return child


def get_alignment(loc):
	with open(f"{loc}.fasta") as f:
		return AlignIO.read(f, 'fasta')


def get_alignment_sequence_dict(alignment):
	dictionary = {}
	for seq in alignment:
		dictionary[seq.name] = seq.seq
	return dictionary


def get_tree(loc):
	with open(f"{loc}.fasta.raxml.bestTree") as f:
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