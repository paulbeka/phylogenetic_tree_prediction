import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, recall_score, balanced_accuracy_score


BASE_SEQUENCES = ['A', 'R', 'I', 'V', 'P', 'S', '-', 'Q', 'D', 'H', 'K', 'Y', 'N', 'L', 'F', 'T', 'C', 'M', 'G', 'E', 'W']


class NodeNetwork(nn.Module):
	def __init__(self, batch_size=1):
		super(NodeNetwork, self).__init__()
		
		self.firstLayer = nn.Linear(22, 30)
		self.secondLayer = nn.Linear(30, 10)
		self.finalLayer = nn.Linear(10, 2)

		self.relu = nn.ReLU()


	def forward(self, x):
		x = self.relu(self.firstLayer(x))
		x = self.relu(self.secondLayer(x))
		x = self.finalLayer(x)

		return x


def train_node_network(dataset, testing_data=None):
	n_epochs = 10
	lr = 0.0005
	batch_size = 3

	dataset = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

	model = NodeNetwork()
	criterion = torch.nn.BCEWithLogitsLoss(weight=torch.Tensor([20, 1]))
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	best_acc = (None, 0)
	for epoch in range(n_epochs):
		for data in dataset:
			optimizer.zero_grad()
			out = model(data[0])
			loss = criterion(out, data[1])
			loss.backward()
			optimizer.step()

		if testing_data:	# Return output depends on testing_data not being None
			balanced_acc = test_node_network(model, testing_data)
			if balanced_acc > best_acc[1]:
				best_acc = (model, balanced_acc)	# MAY NEED TO COPY THE MODEL
				print(f"Balanced accuracy: {balanced_acc:.2f}%")

		print(f'Epoch: {epoch}, Loss: {loss}')

	return best_acc[0]


def test_node_network(model, data):
	predicted_labels = []
	true_labels = []

	with torch.no_grad():
	    for item in data:
	        out = model(item[0])
	        predicted_label = torch.argmax(out)
	        true_label = torch.argmax(item[1])
	        predicted_labels.append(predicted_label)
	        true_labels.append(true_label)

	balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)*100
	# print(confusion_matrix(true_labels, predicted_labels))
		
	return balanced_acc


def load_node_data(tree, original_point=None, generate_true_ratio=False):
	nodes = []
	data = None

	queue = [node for node in tree.tree.find_clades(terminal=True)]
	waiting_parents = set({})
	done = []
	payload_dict = {}
	while queue:
		curr = queue[0]
		payload = {}

		if curr.is_terminal():
			payload = get_amino_acid_frequency(tree.alignment[curr.name])

		else:
			payload = combine_dicts(
				payload_dict[curr.clades[0].name],
				payload_dict[curr.clades[1].name]
			)

		parent = get_parent(tree.tree, curr)
		if parent not in queue and parent not in done:
			if parent in waiting_parents:
				queue.append(parent)
			else:
				waiting_parents.add(parent)

		payload_dict[curr.name] = payload

		dat = {x: (payload[x]/payload["total"]) if x in payload else 0 for x in BASE_SEQUENCES}
		dat = torch.tensor([*list(dat.values()), (get_node_depth(tree.tree, curr)/10)])*10 # add the depth measure
		if original_point == curr:
			data = (dat, torch.Tensor([1, 0]))
		else:
			nodes.append((curr, {"x": dat}))


		done.append(curr)
		queue.remove(curr)

	if generate_true_ratio:
		d = [data]
		d += [(node[1]["x"], torch.Tensor([0, 1])) for node in nodes]
		data = d
	else:
		selected = random.choice(nodes)[1]["x"]
		data = [(selected, torch.Tensor([0, 1])), data]

	return data


def cv_validation_node(dataset):
	acc = []
	for i in range(5):
		train = dataset[:int((i*0.2)*len(dataset))] + dataset[int(((i*0.2)+0.2)*len(dataset)):]
		test = dataset[int((i*0.2)*len(dataset)):int(((i*0.2)+0.2)*len(dataset))]

		model = train_node_network(train, testing_data=test)
		acc.append(test_node_network(model, test))

	return acc


### UTILITY ###
def get_amino_acid_frequency(sequence):
	data = {"total": 0}
	for acid in sequence:
		data["total"] += 1
		try:
			data[acid] += 1
		except:
			data[acid] = 1
	return data
#	return {key: val/data["total"] for (key, val) in data if key != "total"}


def combine_dicts(A, B):
	return {x: A.get(x, 0) + B.get(x, 0) for x in set(A).union(B)}


def get_parent(tree, child_clade):
	node_path = tree.get_path(child_clade)
	try:
		return node_path[-2]
	except:
		return tree.root


def get_node_depth(tree, node):
	node_path = tree.get_path(node)
	return len(node_path)