from get_tree_features import get_tree_features
from util.tree_manager import Tree, randomize_tree
from util.raxml_util import calculate_raxml
from networks.spr_likelihood_prediction_trainer import get_dataloader, train_value_network, test_value_network, test_model_ll_increase
from networks.gnn_network import load_tree, train_gnn_network

import random, dendropy, os, argparse, pickle
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt


TRAIN_TEST_SPLIT = 0.8
BASE_DIR = os.getcwd()
WINDOWS = False


def data_generation(args, returnData=False):
	data_files = find_data_files(os.path.join(BASE_DIR, args.location))[0:4]

	training_data = {
		"spr": [],
		"gnn": []
	}

	for i in tqdm(range(len(data_files))):
		tree = Tree(data_files[i]) 
		dataset, gnn_dataset, base_ll = create_dataset(tree)
		training_data["spr"] += dataset
		training_data["gnn"] += gnn_dataset

	if args.output_dest:
		with open(f"{args.output_dest}/data_generation_output.pickle", "wb") as f:
			pickle.dump(training_data, f)

	else:
		with open(f"data_generation_output.pickle", "wb") as f:
			pickle.dump(training_data, f)

	if returnData:
		return training_data


def train(args):
	training_data = None # load the training data here

	spr_model = train_value_network(training_data["spr"])
	gnn_model = train_gnn_network(training_data["gnn"])

	torch.save(spr_model.state_dict(), f"{args.save_path}/spr_model")
	torch.save(gnn_model.state_dict(), f"{args.save_path}/gnn_model")


def test(args, models=None):
	random_tree = randomize_tree(tree)

	if models[0]:
		test_model_ll_increase(models[0], random_tree)
	if models[1]:
		test_gnn_network(models[1], random_tree)
		


def complete(args):
	training_data = data_generation(args, returnData=True)

	# training_data["spr"] = get_dataloader(training_data["spr"])
	
	# spr_model = train_value_network(training_data["spr"])
	gnn_model = train_gnn_network(training_data["gnn"])

	test(args, models=(None, gnn_model))


def create_dataset(tree, n_items=250, rapid=True):
	dataset = []
	gnn_dataset = []
	base_ll = []
	prev_ll = None
	for i in range(n_items):
		
		actionSpace = tree.find_action_space()
		
		if rapid:
			action = random.choice(actionSpace)
		else:
			best = None
			for action in actionSpace:
				raxml_score = float(callable(tree)["ll"])
				if best and best[1] < raxml_score:
					best = (action, raxml_score)
				else:
					best = (action, raxml_score)
			action = best[0]


		original_point = tree.perform_spr(action[0], action[1], return_parent=True)
		treeProperties = get_tree_features(tree, action[0], original_point)

		# rest = 0? --> ask supervisor
		gnn_data = load_tree(tree, original_point=original_point)
		gnn_dataset.append(gnn_data)

		if WINDOWS:
			raxml_score = 0  # raxml does not work on windows 
		else:
			raxml_score = float(calculate_raxml(tree)["ll"])
		
		base_ll.append((i, raxml_score))

		if not prev_ll:
			score = abs(raxml_score)
		else:
			score = prev_ll - abs(raxml_score)

		dataset.append((treeProperties, score))

	return dataset, gnn_dataset, base_ll


def find_data_files(path):
	list_of_files = set([])
	for file in os.scandir(path):
		if file.name.split(".")[-1] == "tree":
			list_of_files.add(f"{path}/{file.name.split('.')[0]}")
	return list(list_of_files)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="PhloGNN",
		description="This is the GNN phylogenetic tree creator.")
	parser.add_argument("-m", "--mode", required=True,
		choices=["data_generation", "train", "test", "complete"])
	parser.add_argument("-l", "--location", required=True)
	parser.add_argument("-o", "--output_dest")
	parser.add_argument("-w", "--windows", action="store_true")
	args = parser.parse_args()

	WINDOWS = args.windows

	try:
		exec(f"{args.mode}(args)")
	except Exception as e:
		print(e)