import os
from Bio import SeqIO


SEQ_FILE = 'data/Compara.110.ncrna_default.nt.fasta'

sequences = list(SeqIO.parse(SEQ_FILE, "fasta"))


if not sequences:
	print("No sequences found.")
else:
	for seq_record in sequences:
	    print(f"ID: {seq_record.id}")
	    print(f"Description: {seq_record.description}")
	    print(f"Sequence length: {len(seq_record.seq)}")
	    print(f"Sequence: {seq_record.seq}\n")