from get_tree_features import get_tree_features
from algorithm import run_algorithm
from util.tree_manager import Tree, randomize_tree
from util.raxml_util import calculate_raxml
from networks.spr_network import SprScoreFinder, get_dataloader, train_value_network, test_value_network, test_model_ll_increase, compare_score, test_top_10
from networks.gnn_network import load_tree, train_gnn_network, test_gnn_network, GCN
from networks.node_network import train_node_network, load_node_data, test_node_network

import random, dendropy, os, argparse, pickle, torch, copy
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import traceback


TRAIN_TEST_SPLIT = 0.8
BASE_DIR = os.getcwd()
WINDOWS = False


def data_generation(args, returnData=False):
	data_files = find_data_files(os.path.join(BASE_DIR, args.location))

	training_data = generate(data_files)

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
	training_data = None

	spr_model = train_value_network(training_data["spr"])
	gnn_model = train_gnn_network(training_data["gnn"])

	torch.save(spr_model.state_dict(), f"{args.output_dest}/spr_model")
	torch.save(gnn_model.state_dict(), f"{args.output_dest}/gnn_model")
		

def complete(args):
	if "dataset" in args:
		with open(args.dataset, "r") as f:
			data = pickle.load(f) 
	else:
		files = find_data_files(os.path.join(BASE_DIR, args.location))
		training_data = generate(files[:16], generate_true_ratio=False)
		testing_data = generate(files[16:], generate_true_ratio=True)

	training_data["spr"] = get_dataloader(training_data["spr"])
	testing_data["spr"] = get_dataloader(testing_data["spr"])

	spr_model = train_value_network(training_data["spr"], test=testing_data["spr"])

	# compare_score(spr_model, testing_data["spr"])

	gnn_model = train_gnn_network(training_data["gnn"], testing_data=testing_data["gnn"]) #testing_data=testing_data
	# node_model = train_node_network(training_data, testing_data=testing_data)

	torch.save(spr_model.state_dict(), f"{args.output_dest}/spr")
	torch.save(gnn_model.state_dict(), f"{args.output_dest}/gnn")


def algorithm(args):
	spr_model = SprScoreFinder(1)
	spr_model.load_state_dict(torch.load(f"{args.networks_location}/spr"))
	spr_model.eval()

	gnn_model = GCN()
	gnn_model.load_state_dict(torch.load(f"{args.networks_location}/gnn"))
	gnn_model.eval()

	try:
		tree = Tree(args.location)
		run_algorithm(tree, spr_model, gnn_model, 50)
	except Exception as e:
		traceback.print_exc()



def test(args, data=None, models=None):
	if data:
		return test_gnn_network(models[1], data)
	else:
		gnn = torch.load(args.location)#

	# need to calculate the actual score at every iteration, similar to data data_generation
	# then find the top 10 at each and see if the gnn or spr network can detect it 
	# calculate the maximum likelihood reached by algorithm, and the path
	# 	-> need a way to caclaulte the true ll for every move
	# calculate time taken, pretty simple. Try for multiple datasets.
	
	n_items_random_walk = 40

	files = find_data_files(os.path.join(BASE_DIR, args.location))
	testing_dataset = generate(files, n_items_random_walk=n_items_random_walk)
	spr_testing_dataset = [testing_data["spr"][i * len(files):(i + 1) * n] for i in range((len(testing_data["spr"]) + len(files) - 1) // len(files) )]
	print(len(spr_testing_dataset))
	test_top_10(spr_testing_dataset)

#################### NON COMMAND EXECUTABLES ####################

def generate(data_files, generate_true_ratio=True, n_items_random_walk=40):
	training_data = {
		"spr": [],
		"gnn": []
	}

	for i in tqdm(range(len(data_files))):
		tree = Tree(data_files[i]) 
		dataset, gnn_dataset, base_ll = create_dataset(tree, generate_true_ratio=generate_true_ratio, n_items_random_walk=n_items_random_walk)
		training_data["spr"] += dataset
		training_data["gnn"] += gnn_1_move(tree)

	return training_data


def create_dataset(tree, 
		n_items=40,  					# Number of random mutations
		rapid=True, 					# Find best mutation at every time step
		generate_true_ratio=True 		# Generate 1-to-1 (even dataset) or the true ratio
	):
	
	dataset = []
	gnn_dataset = []
	base_ll = []
	prev_ll = None
	for i in range(n_items):
		
		actionSpace = tree.find_action_space()
		if rapid:
			action = random.choice(actionSpace)
			original_point = tree.perform_spr(action[0], action[1], return_parent=True)
			treeProperties = get_tree_features(tree, action[0], original_point)

			# node_data = load_node_data(tree, original_point=original_point, generate_true_ratio=generate_true_ratio)
			# gnn_dataset += node_data
			gnn_data = load_tree(tree, target=[action[0]])
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


		else:
			ranking = []
			for action in actionSpace:
				treeCopy = copy.deepcopy(tree)
				actionCopy = copy.deepcopy(action)
				original_point = treeCopy.perform_spr(actionCopy[0], actionCopy[1], return_parent=True)
				# raxml_score = float(calculate_raxml(treeCopy)["ll"])
				raxml_score = 0
				ranking.append((action, raxml_score))

			ranking.sort(key=lambda x: x[1])
			# gnn_dataset.append((tree, ranking))
			subtr, regraft = ranking[0][0]
			original_point = tree.perform_spr(subtr, regraft, return_parent=True)
			treeProperties = get_tree_features(tree, subtr, original_point)
			dataset.append((treeProperties, ranking[0][1]))

	return dataset, gnn_dataset, base_ll


def gnn_1_move(tree):
	actionSpace = tree.find_action_space()
	random.shuffle(actionSpace)
	already_done = set()
	p = []
	for action in actionSpace:
		if not ((action[0] in already_done) or (action[1] in already_done)):
			p.append(action)
			already_done.add(action[0])
			already_done.add(action[1])
	actionSpace = p

	gnn_dataset = []
	targets = []
	for action in actionSpace[:int(len(actionSpace)/4)]:
		original_point = tree.perform_spr(action[0], action[1], return_parent=True)
		targets.append(original_point)
	gnn_dataset.append(load_tree(tree, target=targets))
	return gnn_dataset


def find_data_files(path):
	list_of_files = set()
	for file in os.scandir(path):
		if file.name.split(".")[-1] == "bestTree":
			list_of_files.add(f"{path}/{file.name.split('.')[0]}")

	return list(list_of_files)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="PhloNN",
		description="This is the Neural Network phylogenetic tree creator.")
	parser.add_argument("-m", "--mode", required=True,
		help="The mode to be used when using the app. Train for training network (and you already have data), test for testing, and complete to do both.",
		choices=["data_generation", "train", "test", "complete", "algorithm"])
	parser.add_argument("-l", "--location", required=True,
		help="Location for the input data (raw tree files to train)")
	parser.add_argument("-o", "--output_dest",
		help="Where the output for the data should be stored.")
	parser.add_argument("-w", "--windows", action="store_true",
		help="If windows is being used, this flag should be true.")
	parser.add_argument("-n", "--networks_location",
		help="Give the location of the stored NNs if algorithm is being run")
	args = parser.parse_args()

	WINDOWS = args.windows

	exec(f"{args.mode}(args)")
