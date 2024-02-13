import torch
import random
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.utils.convert import from_networkx
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from collections import Counter
import torch.nn.functional as F


BASE_SEQUENCES = ['A', 'R', 'I', 'V', 'P', 'S', '-', 'Q', 'D', 'H', 'K', 'Y', 'N', 'L', 'F', 'T', 'C', 'M', 'G', 'E', 'W']


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=32):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(21, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_gnn_network(dataset, testing_data=None):
	n_epochs = 3000
	lr = 0.0001

	model = GCN()
	criterion = torch.nn.BCEWithLogitsLoss(weight=torch.Tensor([40, 1]))
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	def train(data):
		optimizer.zero_grad()
		out = model(data.x.float(), data.edge_index)
		loss = criterion(out, data.y.float())
		loss.backward()
		optimizer.step()
		return loss


	best_acc = 0
	steps_before_test, max_n_tries = 300, 10
	curr, n_tries = 0, 0
	for epoch in range(n_epochs):
		for data in dataset:
			loss = train(data)
			curr += 1
			if testing_data and curr > steps_before_test:
				acc = test_gnn_network(model, testing_data)
				if acc > best_acc:
					best_acc = acc
				
					if n_tries < max_n_tries:
						n_tries += 1
					else:
						return model

				curr = 0

		print(f'Epoch: {epoch}, Loss: {loss}')

	print(f"Best accuracy found: {best_acc:.2f}%")

	return model


# TODO: OPTIMIZE THIS CODE 
def load_tree(tree, 
	target=None, 		# Which node is most "out of place"
	score_correct=20):	# Score for correct node (out of place node)

	G = nx.Graph()

	nodes = []
	edges = []
	attrs = {}

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
			edges.append((curr, curr.clades[0]))
			edges.append((curr, curr.clades[1]))
			attrs[(curr, curr.clades[0])] = {"length": curr.clades[0].branch_length}
			attrs[(curr, curr.clades[1])] = {"length": curr.clades[1].branch_length}

		parent = get_parent(tree.tree, curr)
		if parent not in queue and parent not in done:
			if parent in waiting_parents:
				queue.append(parent)
			else:
				waiting_parents.add(parent)

		payload_dict[curr.name] = payload

		dat = {x: (payload[x]/payload["total"]) if x in payload else 0 for x in BASE_SEQUENCES}
		dat = torch.tensor(list(dat.values()))
		if target == curr:
			nodes.append((curr, {"x": dat, "y": torch.Tensor([1, 0])}))
		else:
			nodes.append((curr, {"x": dat, "y": torch.Tensor([0, 1])}))

		done.append(curr)
		queue.remove(curr)


	G.add_nodes_from(nodes)
	G.add_edges_from(edges)
	nx.set_edge_attributes(G, attrs)

	return from_networkx(G)


def get_amino_acid_frequency(sequence):
	data = {"total": 0}
	for acid in sequence:
		data["total"] += 1
		try:
			data[acid] += 1
		except:
			data[acid] = 1
	return data


def test_gnn_network(model, data):
	n_correct = 0
	n_incorrect = 0

	with torch.no_grad():
		for item in data:
			out = model(item.x.float(), item.edge_index)
			preds = torch.argmax(out, axis=1)
			y = torch.argmax(item.y, axis=1)
			actual = torch.argmin(y).item()

			if preds[actual].item() == 0:
				n_correct += 1

			for i in range(len(list(preds))):
				if preds[i].item() != y[i].item():
					n_incorrect += 1

	accuracy = (n_correct/len(data))*100
	wrong = (n_incorrect/(len(data)*len(list(y))))*100
	print(f"Percent of true labels predicted true: {accuracy:.2f}%")
	print(f"Percent of predictions wrong in general: {wrong:.2f}%")

	return accuracy


### UTILITY CLASSES ###
def visualize_graph(G):
    plt.figure(figsize=(20,20))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, cmap="Set2")
    plt.show()


def combine_dicts(A, B):
	return {x: A.get(x, 0) + B.get(x, 0) for x in set(A).union(B)}


def get_parent(tree, child_clade):
	node_path = tree.get_path(child_clade)
	try:
		return node_path[-2]
	except:
		return tree.root