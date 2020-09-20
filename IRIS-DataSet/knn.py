# /*
# ////////////////////////////////////////////////////////////////////////
# * Luiz Felipe Raveduti Zafiro - RA: 120513
# * Artificial Intelligence - Federal University of São Paulo (SJC)
# * KNN Algorithm for IRIS DataSet
# ////////////////////////////////////////////////////////////////////////
# */


import numpy as np
import random
import math


# Sets randomly 75% of the dataset to training and 25% for testing
# Format expected: line = 'value,value,value,value,class'
# Norm = normalizaton
def prepareDataSet(f_name, norm=True):

	train_data = []
	test_data = []
	test_data_result = []

	# Reading and oppening file (data set)
	with open(f_name, 'r') as f:
		# Iterates line by line in the archive
		for line in f:
			# Train list (75% of the data set)
			if random.random() < 0.75:
				# Creates a list cointaining the separed values
				train_data.append([i.strip() for i in line.split(',')])
			else:
				# Creates a list cointaining the separed values
				test_data.append([i.strip() for i in line.split(',')])

	if norm:
	# Data normalization 
		# Stores the minimum and maximum for each atribute
		min_train = np.zeros(4)
		max_train = np.zeros(4)
		min_test = np.zeros(4)
		max_test = np.zeros(4)
		for i in range(4):
			max_train[i] = max_test[i] = 0.0
			min_train[i] = min_test[i] = 100.0 
	
		# Finds the minimum and maximum values of each atributes
		for i in train_data:
			if i[0] != '':
				for j in range(0,4):
					# Finds a smaller one
					if float(i[j]) < min_train[j]:
						min_train[j] = float(i[j])
					# Finds a smaller one
					elif float(i[j]) > max_train[j]:
						max_train[j] = float(i[j])
	
		for i in test_data:
			if i[0] != '':
				for j in range(0,4):
					# Finds a smaller one
					if float(i[j]) < min_test[j]:
						min_test[j] = float(i[j])
					# Finds a smaller one
					elif float(i[j]) > max_test[j]:
						max_test[j] = float(i[j])			
		
		# Normalize: x = (x - xmin)/(xmax - xmin) 
		for i in train_data:
			if i[0] != '':
				for j in range(0,4):
					i[j] = (float(i[j]) - min_train[j]) / (max_train[j] - min_train[j])
	
		for i in test_data:
			if i[0] != '':
				for j in range(0,4):
					i[j] = (float(i[j]) - min_test[j]) / (max_test[j] - min_test[j])

	# Remove '' string
	for i in range(len(test_data)):
		if test_data[i] == ['']:
			test_data.pop(i)

	# Remove '' string
	for i in range(len(train_data)):
		if train_data[i] == ['']:
			train_data.pop(i)

	# Separetes de test data from the results
	for i in range(len(test_data)):
		test_data_result.append(test_data[i].pop())
	
	return train_data, test_data, test_data_result 


# Calculates the Euclidian disntance of the test to all training instances
def euclidianDist(test, train_data):
	
	dist = []
	# Iterates thru all train tests
	for i in train_data:
		if i[0] != '':
			d = 0
			for j in range(0,4):
				d += (float(i[j]) - float(test[j]))**2

			dist.append(math.sqrt(d))

	return dist


# Returns the k smallest indexes of a np array
def kSmallest(dist, k):
    return sorted(range(len(dist)), key = lambda sub: dist[sub])[:k]


# Find the number of neighbours of each type, in the answer trained
def answTrained(smallest, train_data, k):

	# [number_setosa, number_versicolor, number_virginica]
	answ = [0,0,0]

	for i in range(0,len(smallest)):
		if train_data[smallest[i]] != '':
			if train_data[smallest[i]][4] == 'Iris-setosa':
				answ[0] += 1
			elif train_data[smallest[i]][4] == 'Iris-versicolor':
				answ[1] += 1
			elif train_data[smallest[i]][4] == 'Iris-virginica':
				answ[1] += 1

	return answ


# Checks if the prediction is correct
def checkCorrect(i, answer_train, test_data_result):

	resul = np.where(answer_train == np.amax(answer_train))
	idx = resul[0][0]

	if idx == 0 and test_data_result[i] == 'Iris-setosa':
		return 1

	elif idx == 1 and test_data_result[i] == 'Iris-versicolor':
		return 1

	elif idx == 2 and test_data_result[i] == 'Iris-virginica':
		return 1

	else: return 0


# K-Nearest Neighbours algorithm
# k = number of neighbours (default = 3)
def knn(train_data, test_data, test_data_result, k=5):

	correct = 0

	# Goes thru all test info
	for i in range(len(test_data)):
		if test_data[i][0] != '':
			# Calculate Euclidian distance of neighbours of element i
			calc = euclidianDist(test_data[i], train_data)
			# Gets the indexes of the k smalest values
			smallest = kSmallest(calc, k)
			# Find the number of neighbours of each type, in the answer trained
			answer_train = answTrained(smallest, train_data, k)
			# Checks if the prediction is correct
			correct += checkCorrect(i, answer_train, test_data_result)

	# Returns the percentage of correctness
	return correct / len(test_data)


# Main function
def main():

	train_data, test_data, test_data_result = prepareDataSet('iris.data')

	# Writes the processed data in .txt files
	with open('train.txt','w') as f:
  		f.writelines("%s\n" % data for data in train_data)
	with open('test.txt','w') as f:
  		f.writelines("%s\n" % data for data in test_data)
	with open('test_result.txt','w') as f:
  		f.writelines("%s\n" % data for data in test_data_result)

	one = knn(train_data, test_data, test_data_result, k=3)
	print(40 * '-')
	print('...:: KNN Algorithm - IRIS DataSet ::...')
	print('\nTest for k = 3: Correctness = {:.2f}%'.format(one * 100))
	print(40 * '-')


if __name__ == '__main__':
	main() 