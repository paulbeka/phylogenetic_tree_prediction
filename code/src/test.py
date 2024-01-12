from sequence_loader import get_tree_and_alignment, get_alignment_sequence_dict
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor, Scorer
from Bio.Phylo import write, read


_, alignment = get_tree_and_alignment("data/fast_tree_dataset/COG527")

calculator = DistanceCalculator('identity')
distMatrix = calculator.get_distance(alignment)
treeConstructor = DistanceTreeConstructor()

original_tree = read("data/fast_tree_dataset/COG527.sim.trim.tree", "newick")
tree = treeConstructor.upgma(distMatrix)


