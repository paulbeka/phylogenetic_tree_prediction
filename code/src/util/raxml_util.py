from subprocess import Popen, PIPE, STDOUT
import re, random


RAXML_NG_SCRIPT = "raxml-ng"


def calculate_raxml(tree):
	msa_file = "./"
	tree_rampath = "/dev/shm/" + str(random.random())  + str(random.random()) + "tree"  # the var is the str: tmp{dir_suffix}

	try:
		with open(tree_rampath, "w") as fpw:
			fpw.write(tree.tree.format("newick"))

		raxmlProcess = Popen([RAXML_NG_SCRIPT, '--evaluate', '--msa', "data/fast_tree_dataset/COG527.fasta", '--opt-branches', 'on', '--opt-model', 'off', '--model', "LG", '--nofiles', '--tree', tree_rampath], 
			stdout=PIPE, stdin=PIPE, stderr=STDOUT)

		raxml_stdout = raxmlProcess.communicate()[0]
		raxml_output = raxml_stdout.decode()
		print(raxml_output)
		result = parse_raxmlNG_content(raxml_output) # for testing
		print(result)
		return result

	except Exception as e:
		print(e)


def parse_raxmlNG_content(content):
	"""
	:return: dictionary with the attributes - string typed. if parameter was not estimated, empty string
	"""
	res_dict = dict.fromkeys(["ll", "pInv", "gamma",
							  "fA", "fC", "fG", "fT",
							  "subAC", "subAG", "subAT", "subCG", "subCT", "subGT",
							  "time"], "")

	# likelihood
	ll_re = re.search("Final LogLikelihood:\s+(.*)", content)
	if ll_re:
		res_dict["ll"] = ll_re.group(1).strip()
	elif re.search("BL opt converged to a worse likelihood score by", content) or re.search("failed", content):   # temp, till next version is available
		ll_ini = re.search("initial LogLikelihood:\s+(.*)", content)
		if ll_ini:
			res_dict["ll"] = ll_ini.group(1).strip()
	else:
		res_dict["ll"] = 'unknown raxml-ng error, check "parse_raxmlNG_content" function'


	# gamma (alpha parameter) and proportion of invariant sites
	gamma_regex = re.search("alpha:\s+(\d+\.?\d*)\s+", content)
	pinv_regex = re.search("P-inv.*:\s+(\d+\.?\d*)", content)
	if gamma_regex:
		res_dict['gamma'] = gamma_regex.group(1).strip()
	if pinv_regex:
		res_dict['pInv'] = pinv_regex.group(1).strip()

	# Nucleotides frequencies
	nucs_freq = re.search("Base frequencies.*?:\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)", content)
	if nucs_freq:
		for i,nuc in enumerate("ACGT"):
			res_dict["f" + nuc] = nucs_freq.group(i+1).strip()

	# substitution frequencies
	subs_freq = re.search("Substitution rates.*:\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)", content)
	if subs_freq:
		for i,nuc_pair in enumerate(["AC", "AG", "AT", "CG", "CT", "GT"]):  # todo: make sure order
			res_dict["sub" + nuc_pair] = subs_freq.group(i+1).strip()

	# Elapsed time of raxml-ng optimization
	rtime = re.search("Elapsed time:\s+(\d+\.?\d*)\s+seconds", content)
	if rtime:
		res_dict["time"] = rtime.group(1).strip()
	else:
		res_dict["time"] = 'no ll opt_no time'

	return res_dict