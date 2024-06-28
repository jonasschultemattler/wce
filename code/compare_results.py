import sys
import csv
import numpy as np


# if __name__ == "__main__":
# 	if len(sys.argv) != 2:
# 		print("enter heuristic result")
# 		exit(1)
# 	res_rand, res_rw, res_as = [], [], []
# 	with open(sys.argv[1]) as file:
# 		for row in csv.DictReader(file, delimiter=';'):
# 			instance = row['file'].split('/')[-1].split('.')[0]
# 			if "BAD COST" in row['verified']:
# 				bad_cost = int(row[None][0])
# 				quality = bad_cost/(int(row['solsize']) - bad_cost)*100
# 				if instance[0] == 'r':
# 					res_rand.append(quality)
# 				if instance[0] == 'w':
# 					res_rw.append(quality)
# 				if instance[0] == 'a':
# 					res_as.append(quality)
# 			else:
# 				if instance[0] == 'r':
# 					res_rand.append(0)
# 				if instance[0] == 'w':
# 					res_rw.append(0)
# 				if instance[0] == 'a':
# 					res_as.append(0)
# 	res_rand = np.array(res_rand)
# 	res_rw = np.array(res_rw)
# 	res_as = np.array(res_as)
# 	print("random")
# 	print(res_rand)
# 	print(np.min(res_rand))
# 	print(np.max(res_rand))
# 	print(np.mean(res_rand))
# 	print("real world")
# 	print(res_rw)
# 	print(np.min(res_rw))
# 	print(np.max(res_rw))
# 	print(np.mean(res_rw))
# 	print("action-seq")
# 	print(res_as)
# 	print(np.min(res_as))
# 	print(np.max(res_as))
# 	print(np.mean(res_as))
# 	print("optimally solved")
# 	opt_solved = len(np.where(res_rand == 0)[0]) + len(np.where(res_rw == 0)[0]) + len(np.where(res_as == 0)[0])
# 	solved = res_rand.shape[0] + res_rw.shape[0] + res_as.shape[0]
# 	print(opt_solved/solved*100)

# if __name__ == "__main__":
# 	if len(sys.argv) != 3:
# 		print("enter two files")
# 		exit(1)
# 	data = {}
# 	with open(sys.argv[1]) as file:
# 		for row in csv.DictReader(file, delimiter=';'):
# 			instance = row['file'].split('/')[-1].split('.')[0]
# 			if "BAD COST" in row['verified']:
# 				bad_cost = int(row[None][0])
# 				quality = bad_cost/(int(row['solsize']) - bad_cost)*100 + 0.01
# 				data.update({instance: [quality, ""]})
# 			else:
# 				data.update({instance: [0.01, ""]})
# 	with open(sys.argv[2]) as file:
# 		for row in csv.DictReader(file, delimiter=';'):
# 			instance = row['file'].split('/')[-1].split('.')[0]
# 			if data.get(instance) is not None:
# 				if "BAD COST" in row['verified']:
# 					bad_cost = int(row[None][0])
# 					quality = bad_cost/(int(row['solsize']) - bad_cost)*100 + 0.01
# 				else:
# 					quality =  + 0.01
# 				data[instance][1] = quality
# 			else:
# 				if "BAD COST" in row['verified']:
# 					bad_cost = int(row[None][0])
# 					quality = bad_cost/(int(row['solsize']) - bad_cost)*100 + 0.01
# 				else:
# 					quality =  + 0.01
# 				data.update({instance: ["", quality]})

# 	with open("experiments/compare_quality.csv", 'w+') as file:
# 		types = {'r': "random", 'w': "real-world", 'a': "action-seq"}
# 		writer = csv.writer(file, delimiter=';')
# 		writer.writerow(["file", "Type", "A_quality", "B_quality"])
# 		for instance, (value_a, value_b) in data.items():
# 			writer.writerow([instance, types[instance[0]], value_a, value_b])


if __name__ == "__main__":
	if len(sys.argv) != 4:
		print("enter two files and parameter to compare")
		exit(1)
	data = {}
	with open(sys.argv[1]) as file:
		for row in csv.DictReader(file, delimiter=';'):
			instance = row['file'].split('/')[-1].split('.')[0]
			data.update({instance: [row[sys.argv[3]], ""]})
	with open(sys.argv[2]) as file:
		for row in csv.DictReader(file, delimiter=';'):
			instance = row['file'].split('/')[-1].split('.')[0]
			if data.get(instance) is not None:
				data[instance][1] = row[sys.argv[3]]
			else:
				data.update({instance: ["", row[sys.argv[3]]]})

	with open("experiments/compare_{}.csv".format(sys.argv[3]), 'w+') as file:
		types = {'r': "random", 'w': "real-world", 'a': "action-seq"}
		writer = csv.writer(file, delimiter=';')
		writer.writerow(["file", "Type", "A{}".format(sys.argv[3]), "B{}".format(sys.argv[3])])
		for instance, (value_a, value_b) in data.items():
			writer.writerow([instance, types[instance[0]], value_a, value_b])


# if __name__ == "__main__":
# 	if len(sys.argv) < 3:
# 		print("enter files")
# 		exit(1)
# 	all_times = []
# 	for arg in sys.argv[1:]:
# 		times = []
# 		with open(arg) as file:
# 			for row in csv.DictReader(file, delimiter=';'):
# 				if row['time'] != "":
# 					times.append(float(row['time']))
# 		all_times.append(sorted(times))
# 	m = len(all_times)
# 	n = max([len(times) for times in all_times])
# 	times_matrix = -np.ones((n, m))
# 	for i, times in enumerate(all_times):
# 		for j, time in enumerate(times):
# 			times_matrix[j][i] = time	

# 	with open("experiments/compare_running_times_cactus.csv", 'w+') as file:
# 		writer = csv.writer(file, delimiter=';')
# 		# writer.writerow(["A-sorted", "B-sorted"])
# 		writer.writerow(["A-sorted", "B-sorted", "C-sorted", "D-sorted"])
# 		for i in range(n):
# 			row = []
# 			for j in range(m):
# 				if times_matrix[i][j] == -1:
# 					row.append("")
# 				else:
# 					row.append(times_matrix[i][j])
# 			writer.writerow(row)

# if __name__ == "__main__":
# 	if len(sys.argv) != 2:
# 		print("enter file")
# 		exit(1)
# 	data = {}
# 	with open(sys.argv[1]) as file:
# 		for row in csv.DictReader(file, delimiter=';'):
# 			instance = row['file'].split('/')[-1].split('.')[0]
# 			n_red = int(row['vertices']) - int(row['n_prime']) if len(row['n_prime']) > 0 else 0
# 			d = int(row['d']) if len(row['d']) > 0 else 0
# 			data.update({instance: [n_red, d, row["time"]]})

# 	with open("../wce-students/rr_kweight_only.csv", 'w+') as file:
# 		# types = {'r': "random", 'w': "real-world", 'a': "action-seq"}
# 		writer = csv.writer(file, delimiter=';')
# 		# writer.writerow(["file", "Type", "A{}".format(sys.argv[3]), "B{}".format(sys.argv[3])])
# 		# for instance, (value_a, value_b) in data.items():
# 		# 	writer.writerow([instance, types[instance[0]], value_a, value_b])
# 		writer.writerow(["file", "n_red", "d", "time"])
# 		for instance, (n_red, d, time) in data.items():
# 			writer.writerow([instance, n_red, d, time])



