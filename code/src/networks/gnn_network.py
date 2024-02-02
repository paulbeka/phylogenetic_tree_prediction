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
        self.conv2 = GCNConv(hidden_channels, 2*hidden_channels)
        self.conv3 = GCNConv(2*hidden_channels, hidden_channels)
        self.linear1 = Linear(hidden_channels, 20)
        self.linear2 = Linear(20, 1)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        return x


def train_gnn_network(dataset):
	n_epochs = 50
	lr = 0.0001

	model = GCN()
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	def train(data):
		optimizer.zero_grad()
		out = model(data.x.float(), data.edge_index)
		loss = criterion(out, data.y.unsqueeze(1).float())
		loss.backward()
		optimizer.step()
		return loss

	for epoch in range(n_epochs):
		for data in dataset:
			loss = train(data)
		print(f'Epoch: {epoch}, Loss: {loss}')

	return model
	

# TODO: OPTIMIZE THIS CODE 
def load_tree(tree, original_point=None):

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
		if original_point == curr:
			nodes.append((curr, {"x": dat, "y": 1}))
		else:
			nodes.append((curr, {"x": dat, "y": 0}))

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
	criterion = torch.nn.MSELoss()
	loss = 0

	n_correct = 0
	for item in data:
		print(item)
		out = model(item.x.float(), item.edge_index)
		pred = torch.argmax(out)
		print(pred)
		loss += criterion(out, data.y.unsqueeze(1).float())
		if item[pred] == 1:
			n_correct += 1

	print(f"Accuracy of: {(n_correct/len(data))*100}")
	print(f"Average loss of: {(loss/len(data))*100}")



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