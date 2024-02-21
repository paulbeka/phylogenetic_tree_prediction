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
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


BASE_SEQUENCES = ['A', 'R', 'I', 'V', 'P', 'S', '-', 'Q', 'D', 'H', 'K', 'Y', 'N', 'L', 'F', 'T', 'C', 'M', 'G', 'E', 'W']


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=32):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(21, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 10)
        self.linear1 = Linear(10, 5)
        self.linear2 = Linear(5, 2)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        return x


def train_gnn_network(dataset, testing_data=None):
	if len(dataset) < 1:
		raise Exception("No training data!")
		
	n_epochs = 200
	lr = 0.0001

	model = GCN()
	criterion = torch.nn.BCEWithLogitsLoss(weight=torch.Tensor([41, 1]))
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	def train(data):
		optimizer.zero_grad()
		out = model(data.x.float(), data.edge_index)
		loss = criterion(out, data.y.float())
		loss.backward()
		optimizer.step()
		return loss


	best_acc = (None, 0)
	for epoch in range(n_epochs):
		for data in dataset:
			loss = train(data)
		if testing_data:
			acc = test_gnn_network(model, testing_data, best=best_acc[1])
			if best_acc[1] < acc.item():
				best_acc = (model, acc) 

		print(f'Epoch: {epoch}, Loss: {loss}')

	print(f"Best accuracy found: {best_acc[1]:.2f}%")

	return best_acc[0]


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
		if curr in target:
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


def test_gnn_network(model, data, best=None):
	predicted_labels = []
	true_labels = []

	with torch.no_grad():
	    for item in data:
	        out = model(item.x.float(), item.edge_index)
	        preds = torch.argmax(out, axis=1)
	        y = torch.argmax(item.y, axis=1)
	        
	        predicted_labels.extend(preds.tolist())
	        true_labels.extend(y.tolist())

	balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)*100

	if balanced_acc.item() > best:
		print(f"Balanced Accuracy: {balanced_acc:.2f}%")
		conf_matrix = confusion_matrix(true_labels, predicted_labels)
		print(conf_matrix)

	return balanced_acc


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