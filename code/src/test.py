from get_tree_features import get_tree_features
from tree_manager import Tree

from subprocess import Popen
from tqdm import tqdm


RAXML_NG_SCRIPT = "raxml-ng"


def main():
	tree = Tree()
	


def calculate_raxml(tree):
	data = dendropy.DnaCharacterMatrix.get(
	    path="pythonidae.nex",
	    schema="nexus")
	rx = raxml.RaxmlRunner()
	tree = rx.estimate_tree(
	        char_matrix=data,
	        raxml_args=["--no-bfgs"])
	print(tree.as_string(schema="newick"))


def create_dataset(tree):
	for i in range(1):
		# make a change in the tree, store its features, calculate the likelihood
		# the change will be a random branch that is regrafted
		# might be worth it to check when the "randomness" reaches a maximum distance?
		# Or maybe I should find the optimal branch, and only train with that
		# it would be more accurate but the training time would be much longer
		# should I make a tree handler?
		pass
	get_tree_features(tree)





# use the dataset to train the value network
def train_value_network():
	pass


if __name__ == "__main__":
	main()