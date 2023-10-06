# should I implement some type of UPGMA here?
class Tree:
	def __init__(self, root):
		self.root = root


	# The simcalculation should be able to take into account nodes and Connectors
	def generateTree(self, list_of_nodes, simCalculation):
		
		currentNode = list_of_nodes[0]
		while list_of_nodes:
			nextItem = max(list_of_nodes, lambda x: simCalculation(node, x))	
			list_of_nodes.append(Connector([node, nextItem]))

			list_of_nodes.remove(node)
			list_of_nodes.remove(nextItem)


	def showTree(self):
		pass


class Connector:
	def __init__(self, children, parent=None):
		self.children = children
		self.parent = parent


	def getLeafNodes(self):
		return_vals = []
		for child in self.children:
			if isinstance(child, Node):
				return_vals.append(child)
		if return_vals:
			return return_vals
		return None


	# Here calculate the average value/encoding somehow
	def calculateAverageValue(self):
		return 1


class Node:
	def __init__(self, sequence):
		self.sequence = sequence
