import os
from tqdm import tqdm
from subprocess import Popen, PIPE, STDOUT


RAXML_NG_SCRIPT = "raxml-ng"


def main():
	files = []
	for file in os.scandir(os.getcwd()):
		if file.name.split(".")[-1] == "fasta":
			files.append(file.name)

	for file in tqdm(files):
		raxmlProcess = Popen([RAXML_NG_SCRIPT, '--msa', file, '--model', "LG"], stdout=PIPE, stdin=PIPE, stderr=STDOUT)

	for file in os.scandir(os.getcwd()):
		splitName = file.name.split(".")
		if not (splitName[-1] == "bestTree" or splitName[-1] == "fasta"):
			os.remove(file.name)

	print("Complete.")

if __name__ == "__main__":
	main()