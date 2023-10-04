import os
from Bio import SeqIO
from Bio import Phylo


def load_files():
	DNA_FILE = 'tree_test.tree'
	SPECIES_FILE = 'data/Compara.110.ncrna_default.nh.emf'

	# TODO: Create a handler for the SPECIES_FILE and create a dict species -> genome

	code_to_species = {}

	with open(SPECIES_FILE) as f:
		for line in f:
			if len(line) > 3 and line[:3] == "SEQ":
				dat = line.split(" ")
				code_to_species[dat[2]] = dat[1]

	test = Phylo.read(DNA_FILE, "newick")
	# sequences = list()
	Phylo.draw_ascii(test)
	
	return (None, None)

	return (code_to_species, sequences)


def main():
	code_to_species, sequences = load_files()
	
	if not sequences:
		print("No sequences found.")
	else:
		for seq_record in sequences:
			print(f"ID: {seq_record.id}")

			name = None
			if seq_record.id in code_to_species:
				name = code_to_species[seq_record.id]

			print(f"Name: {name}")
			print(f"Description: {seq_record.description}")
			print(f"Sequence length: {len(seq_record.seq)}")
			print(f"Sequence: {seq_record.seq}\n")


if __name__ == "__main__":
	main()