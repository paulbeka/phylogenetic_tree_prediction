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


# Or maybe I should find the optimal branch, and only train with that
# it would be more accurate but the training time would be much longer
def create_dataset(tree):
	dataset = []
	for i in range(1):
		actionSpace = tree.find_action_space()
		action = random.choice(possible_actions)
		currentTree = tree.perform_spr(action[0], action[1])
		treeProperties = get_tree_features(tree)
		score = calculate_raxml(treeProperties)
		dataset.append((treeProperties, score))

	return dataset


# use the dataset to train the value network
def train_value_network():
	pass


if __name__ == "__main__":
	main()