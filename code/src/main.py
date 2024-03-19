from get_tree_features import get_tree_features
from algorithm import run_algorithm, test_algorithm
from util.tree_manager import Tree, randomize_tree
from util.raxml_util import calculate_raxml
from networks.spr_network import SprScoreFinder, get_dataloader, train_value_network, test_value_network, test_model_ll_increase, compare_score, test_top_10, optimize_spr_network
from networks.gnn_network import load_tree, train_gnn_network, test_gnn_network, GCN, gnn_test_top_10, cv_validation_gnn, optimize_gnn_network, train_until_max_found
from networks.node_network import train_node_network, load_node_data, test_node_network, cv_validation_node, NodeNetwork, optimize_node_network

import random, dendropy, os, argparse, pickle, torch, copy, time
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import traceback
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


TRAIN_TEST_SPLIT = 0.8
BASE_DIR = os.getcwd()
WINDOWS = False


def data_generation(args, returnData=False):
	data_files = find_data_files(os.path.join(BASE_DIR, args.location))

	# MAKE SURE IT CAN GENERATE GNN, NODE, AND SPR DATASETS B4 BIG PROCESSING
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
	files = find_data_files(os.path.join(BASE_DIR, args.location))
	training_data = generate(files[:16], generate_true_ratio=False, generate_node=True)
	testing_data = generate(files[16:], generate_true_ratio=False, generate_node=True)

	spr_model = train_value_network(training_data["spr"], testing_data["spr"])
	gnn_model = train_until_max_found(training_data["gnn"], testing_data["gnn"])

	torch.save(spr_model.state_dict(), f"{args.output_dest}/spr_model")
	torch.save(gnn_model.state_dict(), f"{args.output_dest}/gnn_model")
		

def complete(args): 	# NOTE: RANDOM WALK GNN GENERATION DOES NOT WORK AT ALL.
	if "dataset" in args:
		with open(args.dataset, "r") as f:
			data = pickle.load(f) 
	else:
		files = find_data_files(os.path.join(BASE_DIR, args.location))
		training_data = generate(files[:16], generate_true_ratio=False, generate_node=True)
		testing_data = generate(files[16:], generate_true_ratio=True, generate_node=True)


	t = [x[1] for x in training_data["base_ll"][0]]
	for i in range(1, 15):
		for j in range(len(t)-1):
			t[j] = (t[j] + training_data["base_ll"][i][j][1]) / 2

	plt.plot(t)
	plt.xlabel("Iteration")
	plt.ylabel("Likelihood")
	plt.title("Likelihood to iteration ratio")
	plt.show()

	training_data["spr"] = get_dataloader(training_data["spr"])
	# testing_data["spr"] = get_dataloader(testing_data["spr"])
	spr_testing_dataset = [testing_data["spr"][i * len(files):(i + 1) * len(files)] for i in range((len(testing_data["spr"]) + len(files) - 1) // len(files) )]

	# NEXT: SWITCH TO THE 1 SHOT GNN AND ALSO MAKE IT LOOP ON THE SAME TREE INSTEAD OF SMALL DATASET
	# THEN, RUN ON LINUX TO FINALLY GET A WORKING SPR NETWORK
	if args.optimize:
		spr_model = optimize_spr_network(training_data["spr"], spr_testing_dataset)
		gnn_model = optimize_gnn_network(training_data["gnn"], testing_data["gnn"])
		node_model = optimize_node_network(training_data["node"], testing_data["node"])

	else:
		spr_model = train_value_network(training_data["spr"], test=spr_testing_dataset).state_dict()
		gnn_model = train_until_max_found(training_data["gnn"], testing_data=testing_data["gnn"]).state_dict()
		node_model = train_node_network(training_data["node"], testing_data=testing_data["node"]).state_dict()

	torch.save(spr_model, f"{args.output_dest}/spr")
	torch.save(gnn_model, f"{args.output_dest}/gnn")
	torch.save(node_model, f"{args.output_dest}/node")


def algorithm(args, testing=False):
	spr_model, gnn_model, node_model = load_models(args)
	try:
		tree = Tree(args.location)
		original_score = calculate_raxml(tree)["ll"]
		tree = shuffle_tree(tree)
		t0 = time.time()
		final_tree = test_algorithm(tree, original_score, spr_model, gnn_model)
		final_time = time.time() - t0
		print(f"Time taken to run: {final_time}")
		return final_time, final_tree
	except Exception as e:
		traceback.print_exc()

	if testing:
		data_files = find_data_files(os.path.join(BASE_DIR, args.location))
		n_times = 10
		original_scores = []
		avg = [[]]*n_times
		times = []
		for file in data_files:
			tree = Tree(file)
			original_scores.append(calculate_raxml(tree)["ll"])
			tree = shuffle_tree(tree, n_times)
			t0 = time.time()
			final_tree, path = run_algorithm(tree, spr_model, gnn_model, find_true_ll_path=False)
			final_time = time.time() - t0
			times.append(final_time)
			print(f"Time taken to run: {final_time}")
			for i, item in enumerate(path):
				avg[i].append(item)

		avg = [sum(avg[i])/len(avg[i]) for i in range(len(avg))]

		print(f"Final average original score: {sum(original_scores)/len(original_scores):.2f}")
		print(f"Final average score: {sum(avg[-1])/len(avg[-1]):.2f}")
		print(f"Average exec time: {sum(times)/len(times):.2f}")

		plt.plot(avg)
		plt.title("Average likelihood over iterations")
		plt.xlabel("Number of iterations")
		plt.ylabel("Average likelihood")
		plt.savefig("alg_output_avg")





# !!!!!!!!!!!! TODO: SPR RAXML-NG TEST BEST MOVE %
def test(args, data=None, models=None):
	# TODAY: CHANGE TO 1 SHOT GNN
	spr_model, gnn_model, node_model = load_models(args)

	if args.data:
		pass # Open the file with the large datasets here
	else:
		n_items_random_walk = 40
		files = find_data_files(os.path.join(BASE_DIR, args.location))
		testing_data = generate(files, n_items_random_walk=n_items_random_walk, generate_node=True, multiple_move=False)
		spr_testing_dataset = [testing_data["spr"][i * len(files):(i + 1) * len(files)] for i in range((len(testing_data["spr"]) + len(files) - 1) // len(files) )]
	
	spr_top_10 = test_top_10(spr_model, spr_testing_dataset)
	gnn_top_10 = gnn_test_top_10(gnn_model, testing_data["gnn"])

	print(f"SPR percentage in top 10: {spr_top_10*100:.2f}%")
	# print(f"GNN percentage in top 10: {gnn_top_10*100:.2f}%")

	gnn_preds, gnn_true = [], []
	node_preds, node_true = [], []
	for i in range(100):
		gnn_item = testing_data["gnn"][i]
		gnn_preds += [x.item() for x in list(gnn_model(gnn_item.x, gnn_item.edge_index)[:, 0])]
		gnn_true += [int(x.item()) for x in list(gnn_item.y[:, 0])]

	for i in range(100):
		item = testing_data["node"][i]
		node_preds += [x.item() for x in list(node_model(item[0]))]
		node_true += [int(x.item()) for x in list(item[1])]

	auc_score_gnn = roc_auc_score(gnn_true, gnn_preds)
	auc_score_node = roc_auc_score(node_true, node_preds)

	print(f"GNN AUC Score: {auc_score_gnn}")
	print(f"Node AUC Score: {auc_score_node}")

	# remake data with multiple move for CV validation
	testing_data = generate(files, n_items_random_walk=n_items_random_walk, generate_node=True, multiple_move=True)
	print(len(testing_data["node"]))

	acc_gnn = cv_validation_gnn(testing_data["gnn"])
	acc_node = cv_validation_node(testing_data["node"])

	print(f"GNN accuracy 5-fold CV: {acc_gnn}")
	print(f"Node accuracy 5-fold CV: {acc_node}")

	gnn_mean, node_mean = sum(acc_gnn)/len(acc_gnn), sum(acc_node)/len(acc_node)
	gnn_err, node_err = max([abs(x-gnn_mean) for x in acc_gnn]), max([abs(x-node_mean) for x in acc_node]) 

	plt.figure(figsize=(8, 6))
	bars = plt.bar(["GNN", "Node"], [gnn_mean, node_mean], yerr=[gnn_err, node_err], color=['blue', 'red'], capsize=5)
	plt.ylabel('Balanced accuracy')
	plt.title('Performance Comparison') 
	plt.xticks(rotation=45)
	plt.ylim(0, 100) 

	for bar in bars:
	    yval = bar.get_height()
	    plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval,2), va='bottom')

	plt.tight_layout()
	plt.show()


#################### NON COMMAND EXECUTABLES ####################

def load_models(args):
	spr_model = SprScoreFinder(1)
	spr_model.load_state_dict(torch.load(f"{args.networks_location}/spr"))
	spr_model.eval()

	gnn_model = GCN()
	gnn_model.load_state_dict(torch.load(f"{args.networks_location}/gnn"))
	gnn_model.eval()

	node_model = NodeNetwork()
	node_model.load_state_dict(torch.load(f"{args.networks_location}/node"))
	node_model.eval()

	return spr_model, gnn_model, node_model


# TODO: add option for gnn 1 move or regular
def generate(data_files, generate_true_ratio=False, n_items_random_walk=40, generate_node=False, multiple_move=True):
	training_data = {
		"spr": [],
		"gnn": [],
		"node":[],
		"base_ll": []
	}
	for i in tqdm(range(len(data_files))):
		tree = Tree(data_files[i]) 
		dataset, gnn_dataset, node_dataset, base_ll = create_dataset(tree, 
			generate_true_ratio=generate_true_ratio, 
			n_items=n_items_random_walk,
		 	generate_node=generate_node)

		training_data["spr"] += dataset
		training_data["node"] += node_dataset
		if multiple_move:
			training_data["gnn"] += gnn_1_move(tree) 	# MAKE IT WORK FOR TESTING
		else:
			training_data["gnn"] += gnn_dataset		# Random walk does not work with GNN
		training_data["base_ll"].append(base_ll)

	return training_data


def create_dataset(tree, 
		n_items=40,  					# Number of random mutations
		rapid=True, 					# Find best mutation at every time step
		generate_true_ratio=True, 		# Generate 1-to-1 (even dataset) or the true ratio
		generate_node=False,			# Generate data for node network
	):
	
	dataset = []
	gnn_dataset = []
	node_dataset = []
	base_ll = []
	for i in range(n_items):
		actionSpace = tree.find_action_space()

		# Get the previous raxml-score
		if WINDOWS:
			prev_raxml_score = 0
		else:
			prev_raxml_score = float(calculate_raxml(tree)["ll"])

		if rapid:
			# CHECK IF THE MOVE HAS INCREASED IN LIKELIHOOD AND IF IT HAS, THEN APPLY IT????
			action = random.choice(actionSpace)
			original_point = tree.perform_spr(action[0], action[1], return_parent=True)
			treeProperties = get_tree_features(tree, action[0], original_point)

			if generate_node:
				node_data = load_node_data(tree, original_point=original_point, generate_true_ratio=generate_true_ratio)
				node_dataset += node_data

			gnn_data = load_tree(tree, target=[action[0]])
			gnn_dataset.append(gnn_data)

			if WINDOWS:
				raxml_score = 0  # raxml does not work on windows 
			else:
				raxml_score = float(calculate_raxml(tree)["ll"])
			
			base_ll.append((i, raxml_score))
			score = prev_raxml_score - raxml_score
			prev_raxml_score = raxml_score

			dataset.append((treeProperties, score))


		else:	# NOTE: NO NODE DATASET IN NON-RAPID MODE. TODO: INTEGRATE THIS 
			ranking = []
			for action in actionSpace:
				treeCopy = copy.deepcopy(tree)
				actionCopy = copy.deepcopy(action)
				original_point = treeCopy.perform_spr(actionCopy[0], actionCopy[1], return_parent=True, deepcopy=True)
				if WINDOWS:
					raxml_score = 0
				else:
					raxml_score = float(calculate_raxml(treeCopy)["ll"])
				ranking.append((action, raxml_score))

			ranking.sort(key=lambda x: x[1])
			subtr, regraft = ranking[0][0]
			original_point = tree.perform_spr(subtr, regraft, return_parent=True)
			gnn_dataset.append(load_tree(tree, target=[subtr]))
			if generate_node:
				node_data = load_node_data(tree, original_point=original_point, generate_true_ratio=generate_true_ratio)
				node_dataset += node_data
			treeProperties = get_tree_features(tree, subtr, original_point)
			dataset.append((treeProperties, ranking[0][1]))

	return dataset, gnn_dataset, node_dataset, base_ll


def shuffle_tree(tree, n_times):
	for i in range(n_times):
		actionSpace = tree.find_action_space()
		action = random.choice(actionSpace)
		tree.perform_spr(action[0], action[1], return_parent=True)
	return tree


def gnn_1_move(tree):
	actionSpace = tree.find_action_space()
	already_done = set()
	p = []
	# START FROM THE BOTTOM NODES AND WORK UP
	for action in actionSpace:
		if not ((action[0] in already_done) or (action[1] in already_done)):
			p.append(action)
			already_done.add(action[0])
			already_done.add(action[1])
	actionSpace = p

	gnn_dataset = []
	targets = []
	for action in actionSpace:
		try:
			original_point = tree.perform_spr(action[0], action[1], return_parent=True)
			targets.append(original_point)
		except:
			pass
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
	parser.add_argument("-d", "--data",
		help="The location of the bulk data to test/train neural networks")
	parser.add_argument("-O", "--optimize", action="store_true",
		help="Optimize the network hyperparameters while running")
	args = parser.parse_args()

	WINDOWS = args.windows

	exec(f"{args.mode}(args)")
