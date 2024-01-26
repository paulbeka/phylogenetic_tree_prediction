import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.utils.convert import from_networkx
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from collections import Counter


BASE_SEQUENCES = ['A', 'R', 'I', 'V', 'P', 'S', '-', 'Q', 'total', 'D', 'H', 'K', 'Y', 'N', 'L', 'F', 'T', 'C', 'M', 'G', 'E', 'W']


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(21, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, 1)

    def forward(self, x, edge_index):
        print(x)
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        
        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h


def train_gnn_network(tree):
	model = GCN()
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

	def train(data):
		optimizer.zero_grad()
		out, h = model(tree.tree.root, 0)
		loss = criterion(0, 0)
		loss.backward()
		optimizer.step()
		return loss, h

	for epoch in range(401):
		data = load_tree(tree)
		print(data)
		print(data.A)
		loss, h = train(data)
		print(f'Epoch: {epoch}, Loss: {loss}')


# TODO: OPTIMIZE THIS CODE 
def load_tree(tree):

	G = nx.Graph()

	nodes = []
	edges = []

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

		parent = get_parent(tree.tree, curr)
		if parent not in queue and parent not in done:
			if parent in waiting_parents:
				queue.append(parent)
			else:
				waiting_parents.add(parent)

		payload_dict[curr.name] = payload

		nodes.append((curr, {x: payload[x] if x in payload else 0 for x in BASE_SEQUENCES}))

		done.append(curr)
		queue.remove(curr)


	G.add_nodes_from(nodes)
	G.add_edges_from(edges)

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


def visualize_graph(G):
    plt.figure(figsize=(20,20))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, cmap="Set2")
    plt.show()



### UTILITY CLASSES ###
def combine_dicts(A, B):
	return {x: A.get(x, 0) + B.get(x, 0) for x in set(A).union(B)}


def get_parent(tree, child_clade):
	node_path = tree.get_path(child_clade)
	try:
		return node_path[-2]
	except:
		return tree.root