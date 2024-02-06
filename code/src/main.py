from get_tree_features import get_tree_features
from util.tree_manager import Tree, randomize_tree
from util.raxml_util import calculate_raxml
from networks.spr_likelihood_prediction_trainer import get_dataloader, train_value_network, test_value_network, test_model_ll_increase
from networks.gnn_network import load_tree, train_gnn_network, test_gnn_network
from networks.node_network import train_node_network, load_node_data, test_node_network

import random, dendropy, os, argparse, pickle, torch
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt


TRAIN_TEST_SPLIT = 0.8
BASE_DIR = os.getcwd()
WINDOWS = False


def data_generation(args, returnData=False, score_correct=None):
	data_files = find_data_files(os.path.join(BASE_DIR, args.location))[0:10]

	training_data = {
		"spr": [],
		"gnn": []
	}

	for i in tqdm(range(len(data_files))):
		tree = Tree(data_files[i]) 
		dataset, gnn_dataset, base_ll = create_dataset(tree, score_correct=score_correct)
		training_data["spr"] += dataset
		training_data["gnn"] += gnn_dataset


	if returnData:
		return training_data
	else:
		if args.output_dest:
			with open(f"{args.output_dest}/data_generation_output.pickle", "wb") as f:
				pickle.dump(training_data, f)
		else:
			with open(f"data_generation_output.pickle", "wb") as f:
				pickle.dump(training_data, f)


def train(args):
	training_data = None # load the training data here

	spr_model = train_value_network(training_data["spr"])
	gnn_model = train_gnn_network(training_data["gnn"])

	torch.save(spr_model.state_dict(), f"{args.output_dest}/spr_model")
	torch.save(gnn_model.state_dict(), f"{args.output_dest}/gnn_model")


def test(args, data=None, models=None):

	if data:
		return test_gnn_network(models[1], data)

	else:
		gnn = torch.load(args.location)

	# random_tree = randomize_tree(tree)

	# if models[0]:
	# 	test_model_ll_increase(models[0], random_tree)
	# if models[1]:
	# 	test_gnn_network(models[1], random_tree)
		

def complete(args):
	# if not args.location:
	# else:
	# 	with open(f"{args.location}/data_generation_output.pickle", "r") as f:
	# 		data = pickle.load(f)

	acc_scores = []
	avg_loss_scores = []

	data = data_generation(args, returnData=True)

	# RETEST WITH GNN AND NOT THIS STUPID [:] BLUNDER
	training_data = data["gnn"][:int(len(data["gnn"])*TRAIN_TEST_SPLIT)]
	testing_data = data["gnn"][int(len(data["gnn"])*TRAIN_TEST_SPLIT):]

	# training_data["spr"] = get_dataloader(training_data["spr"])
	
	# spr_model = train_value_network(training_data["spr"])
	# gnn_model = train_gnn_network(training_data)
	node_model = train_node_network(training_data)

	# torch.save(spr_model.state_dict(), f"{args.output_dest}/spr_model")
	# torch.save(gnn_model.state_dict(), f"{args.output_dest}/gnn_model")

	# acc, loss = test(args, data=testing_data, models=(None, gnn_model))
	# acc_scores.append(acc)
	# avg_loss_scores.append(loss)

	test_node_network(node_model, testing_data)


	print(acc_scores, avg_loss_scores)
	plt.plot(acc_scores)
	plt.show()
	plt.plot([loss/test_score_correct[i] for i, loss in enumerate(avg_loss_scores)])
	plt.show()
		

def create_dataset(tree, n_items=250, rapid=True, score_correct=None):
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

		node_data = load_node_data(tree, original_point=original_point)
		gnn_dataset += node_data
		# gnn_data = load_tree(tree, original_point=original_point, score_correct=score_correct)
		# gnn_dataset.append(gnn_data)

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
		prog="PhloNN",
		description="This is the Neural Network phylogenetic tree creator.")
	parser.add_argument("-m", "--mode", required=True,
		help="The mode to be used when using the app. Train for training network (and you already have data), test for testing, and complete to do both.",
		choices=["data_generation", "train", "test", "complete"])
	parser.add_argument("-l", "--location", required=True,
		help="Location for the input data")
	parser.add_argument("-o", "--output_dest",
		help="Where the output for the data should be stored.")
	parser.add_argument("-w", "--windows", action="store_true",
		help="If windows is being used, this flag should be true.")
	args = parser.parse_args()

	WINDOWS = args.windows

	exec(f"{args.mode}(args)")
