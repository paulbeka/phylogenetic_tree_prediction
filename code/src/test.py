from get_tree_features import get_tree_features
from tree_manager import Tree

from subprocess import Popen, PIPE, STDOUT
from tqdm import tqdm
import random
import dendropy


RAXML_NG_SCRIPT = "raxml-ng"


def main():
	tree = Tree()
	create_dataset(tree, 1)


def calculate_raxml(tree):
	msa_file = "./"
	tree_rampath = "/dev/shm/" + str(random.random())  + str(random.random()) + "tree"  # the var is the str: tmp{dir_suffix}

	try:
		with open(tree_rampath, "w") as fpw:
			fpw.write(tree.tree.format("newick"))

		raxmlProcess = Popen([RAXML_NG_SCRIPT, '--evaluate', '--msa', "data/fast_tree_dataset/COG527.fasta", '--opt-branches', 'on', '--opt-model', 'off', '--model', "LG", '--nofiles', '--tree', tree_rampath], 
			stdout=PIPE, stdin=PIPE, stderr=STDOUT)

		raxml_stdout = raxmlProcess.communicate()[0]
		raxml_output = raxml_stdout.decode()

		print(raxml_output) # for testing
		
		return raxml_output

	except Exception as e:
		print(e)


# Or maybe I should find the optimal branch, and only train with that
# it would be more accurate but the training time would be much longer
def create_dataset(tree, n_items):
	dataset = []
	for i in range(n_items):
		actionSpace = tree.find_action_space()
		action = random.choice(actionSpace)
		# tree.perform_spr(action[0], action[1])
		treeProperties = get_tree_features(tree)
		score = calculate_raxml(tree)
		dataset.append((treeProperties, score))

	return dataset


# use the dataset to train the value network
def train_value_network():
	pass


if __name__ == "__main__":
	main()