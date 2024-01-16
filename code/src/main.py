from get_tree_features import get_tree_features
from tree_manager import Tree
from util.raxml_util import calculate_raxml

from tqdm import tqdm
import random
import dendropy


def main():
	tree = Tree()
	create_dataset(tree, 1)


# Or maybe I should find the optimal branch, and only train with that
# it would be more accurate but the training time would be much longer
def create_dataset(tree, n_items):
	dataset = []
	for i in range(n_items):
		actionSpace = tree.find_action_space()
		action = random.choice(actionSpace)
		# TODO: SPR operation not working --> not being parsed by RAxML-ng
		tree.perform_spr(action[0], action[1])
		treeProperties = get_tree_features(tree)
		score = calculate_raxml(tree)
		dataset.append((treeProperties, score))

	return dataset


# use the dataset to train the value network
def train_value_network():
	pass


if __name__ == "__main__":
	main()