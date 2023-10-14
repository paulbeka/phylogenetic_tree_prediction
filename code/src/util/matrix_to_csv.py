import pandas as pd
import numpy as np
import csv


def toCSV(path, filename):
	data = []
	with open(f'{path}/{filename}', 'r', newline='') as f:
		next(f)
		for line in f:
			line_data = list(filter(None, line.split("\t")))
			data.append([line_data[-1].replace("\r", "").replace("\n", "")])
			data[-1] += [0] * (100 - (len(line_data)-1))
			data[-1] += [float(item.replace(" ", "")) for item in line_data[:-1]]

		arr = [list(x) for x in list(np.array(data))]
		for a in arr:
			if " " in a[0]:
				a[0] = a[0][:3] + a[0].split(" ")[1]
			if "_" in a[0]:
				a[0] = a[0][:3] + a[0].split("_")[1]

		arr.insert(0, [""] + [a[0] for a in arr])

		assert len([a[0] for a in arr]) == len(set([a[0] for a in arr]))

		for i in range(1, len(arr)):
			for j in range(1, len(arr[i])):
				arr[j][i] = arr[i][j]

		with open(f'{path}/{filename}.csv', 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for row in arr:
				writer.writerow(row)


if __name__ == '__main__':
	toCSV("../data/testing", "output_predefined_tree")