import torch
import torch.nn as nn
import torch.nn.functional as F
import random


BASE_SEQUENCES = ['A', 'R', 'I', 'V', 'P', 'S', '-', 'Q', 'D', 'H', 'K', 'Y', 'N', 'L', 'F', 'T', 'C', 'M', 'G', 'E', 'W']


class NodeNetwork(nn.Module):
	def __init__(self, batch_size=1):
		super(NodeNetwork, self).__init__()
		
		self.firstLayer = nn.Linear(21, 30)
		self.secondLayer = nn.Linear(30, 10)
		self.finalLayer = nn.Linear(10, 2)

		self.relu = nn.ReLU()


	def forward(self, x):
		x = self.relu(self.firstLayer(x))
		x = self.relu(self.secondLayer(x))
		x = self.finalLayer(x)

		return x


def train_node_network(dataset, testing_data=None):
	n_epochs = 5
	lr = 0.001

	model = NodeNetwork()
	criterion = torch.nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	prev_acc = 0

	steps_before_test, max_n_tries = 10000, 10
	curr, n_tries = 0, 0
	print(len(dataset))
	for epoch in range(n_epochs):
		for data in dataset:
			optimizer.zero_grad()
			out = model(data[0])
			loss = criterion(out, data[1])
			loss.backward()
			optimizer.step()

			curr += 1

			if testing_data and curr > steps_before_test:
				acc = test_node_network(model, testing_data)
				if acc > prev_acc:
					prev_acc = acc
				
					if n_tries < max_n_tries:
						n_tries += 1
					else:
						return model

				steps_before_test = 0

				print(f'Epoch: {epoch}, Loss: {loss}')

	return model


def test_node_network(model, data):
	n_correct = 0

	with torch.no_grad():
		for item in data:
			out = model(item[0])
			if torch.argmax(out) == torch.argmax(item[1]):
				n_correct += 1

	accuracy = (n_correct/len(data))*100
	print(f"Accuracy of {accuracy:.2f}%")
	
	return accuracy


def load_node_data(tree, original_point=None):
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
		dat = torch.tensor(list(dat.values()))
		if original_point == curr:
			nodes.append((curr, {"x": dat}))
		else:
			data = (dat, torch.Tensor([1, 0]))

		done.append(curr)
		queue.remove(curr)

	selected = random.choice(nodes)[1]["x"]

	data = [(selected, torch.Tensor([0, 1])), data]

	return data


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


def combine_dicts(A, B):
	return {x: A.get(x, 0) + B.get(x, 0) for x in set(A).union(B)}


def get_parent(tree, child_clade):
	node_path = tree.get_path(child_clade)
	try:
		return node_path[-2]
	except:
		return tree.root