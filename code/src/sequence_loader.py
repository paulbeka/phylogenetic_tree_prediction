import dendropy


def main():
	with open('data/testing/distance_matrix.csv') as file:
		pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(
			file,
			is_first_row_column_names=True,
            is_first_column_row_names=True,
            is_allow_new_taxa=True,
            delimiter=",",
		)

	nj_tree = pdm.nj_tree()




# def load_files():
# 	DNA_FILE = 'data/Compara.110.ncrna_default.nt.fasta'
# 	SPECIES_FILE = 'data/Compara.110.ncrna_default.nh.emf'

# 	# TODO: Create a handler for the SPECIES_FILE and create a dict species -> genome

# 	code_to_species = {}

# 	with open(SPECIES_FILE) as f:
# 		for line in f:
# 			if len(line) > 3 and line[:3] == "SEQ":
# 				dat = line.split(" ")
# 				code_to_species[dat[2]] = dat[1]

# 	sequences = list(SeqIO.parse(DNA_FILE, "fasta"))

# 	return (code_to_species, sequences)


# def main():
# 	code_to_species, sequences = load_files()
	
# 	if not sequences:
# 		print("No sequences found.")
# 	else:
# 		for seq_record in sequences:
# 			print(f"ID: {seq_record.id}")

# 			name = None
# 			if seq_record.id in code_to_species:
# 				name = code_to_species[seq_record.id]

# 			print(f"Name: {name}")
# 			print(f"Description: {seq_record.description}")
# 			print(f"Sequence length: {len(seq_record.seq)}")
# 			print(f"Sequence: {seq_record.seq}\n")


if __name__ == "__main__":
	main()