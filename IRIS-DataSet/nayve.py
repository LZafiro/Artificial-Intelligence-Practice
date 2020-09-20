# /*
# ////////////////////////////////////////////////////////////////////////
# * Luiz Felipe Raveduti Zafiro - RA: 120513
# * Artificial Intelligence - Federal University of São Paulo (SJC)
# * Nayve Bayes Algorithm for IRIS DataSet
# ////////////////////////////////////////////////////////////////////////
# */


import numpy as np
import random
import math


# Sets randomly 75% of the dataset to training and 25% for testing
# Format expected: line = 'value,value,value,value,class'
# Norm = normalizaton
def prepareDataSet(f_name):

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


# Calculates the mean and standard deviation of given train dataset
# Returns list, for each type: [(mean1,sd1),(mean2,sd2),(mean3,sd3),(mean4,sd4)]
def atributeProb(train_data):
	# Stores the number of each atribute in each of the classes
	set_cont = 0
	ver_cont = 0
	vir_cont = 0
	# Stores the total value of each atribute of each of the classes
	set_val = np.zeros(4, dtype=float)
	ver_val = np.zeros(4, dtype=float)
	vir_val = np.zeros(4, dtype=float)

	# Calculates the number of each type and the acumulated sum
	for i in train_data:
		if i[4] == 'Iris-setosa':
			set_cont += 1
			set_val[0] += i[0]
			set_val[1] += i[1]
			set_val[2] += i[2]
			set_val[3] += i[3]

		elif i[4] == 'Iris-versicolor':
			ver_cont += 1
			ver_val[0] += i[0]
			ver_val[1] += i[1]
			ver_val[2] += i[2]
			ver_val[3] += i[3]

		elif i[4] == 'Iris-virginica':
			vir_cont += 1
			vir_val[0] += i[0]
			vir_val[1] += i[1]
			vir_val[2] += i[2]
			vir_val[3] += i[3]

	# Calculates each mean
	set_mean = np.zeros(4, dtype=float)
	ver_mean = np.zeros(4, dtype=float)
	vir_mean = np.zeros(4, dtype=float)
	for i in range(4):
		set_mean[i] = set_val[i] / set_cont
	for i in range(4):
		ver_mean[i] = ver_val[i] / ver_cont
	for i in range(4):
		vir_mean[i] = vir_val[i] / vir_cont
	del set_val, ver_val, vir_val

	set_sd = np.zeros(4, dtype=float)
	ver_sd = np.zeros(4, dtype=float)
	vir_sd = np.zeros(4, dtype=float)
	# Calculates the standard deviation (squared)
	for i in train_data:
		if i[4] == 'Iris-setosa':
			set_sd[0] += ( i[0] - set_mean[0] ) ** 2
			set_sd[1] += ( i[1] - set_mean[1] ) ** 2
			set_sd[2] += ( i[2] - set_mean[2] ) ** 2
			set_sd[3] += ( i[3] - set_mean[3] ) ** 2

		elif i[4] == 'Iris-versicolor':
			ver_sd[0] += ( i[0] - ver_mean[0] ) ** 2
			ver_sd[1] += ( i[1] - ver_mean[1] ) ** 2
			ver_sd[2] += ( i[2] - ver_mean[2] ) ** 2
			ver_sd[3] += ( i[3] - ver_mean[3] ) ** 2

		elif i[4] == 'Iris-virginica':
			vir_sd[0] += ( i[0] - vir_mean[0] ) ** 2
			vir_sd[1] += ( i[1] - vir_mean[1] ) ** 2
			vir_sd[2] += ( i[2] - vir_mean[2] ) ** 2
			vir_sd[3] += ( i[3] - vir_mean[3] ) ** 2
	# Dividing each sd by the quantity
	for i in range(4):
		set_sd[i] = set_sd[i] / set_cont
	for i in range(4):
		ver_sd[i] = ver_sd[i] / ver_cont
	for i in range(4):
		vir_sd[i] = vir_sd[i] / vir_cont

	set_out = []
	ver_out = []
	vir_out = []
	# Generates output
	for i in range(4):
		s = []
		s.append(set_sd[i])
		s.append(set_mean[i])
		set_out.append(s)
		s = []
		s.append(ver_sd[i])
		s.append(ver_mean[i])
		ver_out.append(s)
		s = []
		s.append(vir_sd[i])
		s.append(vir_mean[i])
		vir_out.append(s)
	del set_sd, set_mean, ver_sd, ver_mean, vir_sd, vir_mean

	# Probability of each class
	prob = []
	prob.append(set_cont/len(train_data))
	prob.append(ver_cont/len(train_data))
	prob.append(vir_cont/len(train_data))

	return set_out, ver_out, vir_out, prob


# Calculates the PDF for a given non discovered atribute, its mean and sd (msd -> mean and sd)
def gaussianProbabilityDensity(x, msd):
	a = 1 / math.sqrt( 2 * math.pi * msd[0] )
	b = math.e ** - ( ( (x - msd[1]) ** 2 ) / (2 * msd[0]) )
	return a * b


# Calculates the probability of, given test atributes, be of each class (nayve Theorem)
def classProb(test_case, prob, set_msd, ver_msd, vir_msd):

	# Setosa class
	for i in range(4):
		if i == 0:
			setosa = gaussianProbabilityDensity(test_case[i], set_msd[i])
		else:
			setosa *= gaussianProbabilityDensity(test_case[i], set_msd[i])
	
	# Veriscolor class
	for i in range(4):
		if i == 0:
			veriscolor = gaussianProbabilityDensity(test_case[i], ver_msd[i])
		else:
			veriscolor *= gaussianProbabilityDensity(test_case[i], ver_msd[i])
	
	# Virginica class
	for i in range(4):
		if i == 0:
			virginica = gaussianProbabilityDensity(test_case[i], vir_msd[i])
		else:
			virginica *= gaussianProbabilityDensity(test_case[i], vir_msd[i])

	# Posterior probability calculus
	post = []
	den = setosa * prob[0] + veriscolor * prob[1] + virginica * prob[2]
	post.append((setosa * prob[0]) / den)
	post.append((veriscolor * prob[1]) / den)
	post.append((virginica * prob[2]) / den)
	
	post = np.array(post)
	resul = np.where(post == np.amax(post))
	idx = resul[0][0]

	if idx == 0: return 'Iris-setosa'
	elif idx == 1: return 'Iris-versicolor'
	elif idx == 2: return 'Iris-virginica'


# Executes all the classification of the test dataset
# Retuns the acurracy percentage
def classification(train_data, test_data, test_data_result):

	count = 0

	# Calculates the pair (sd, mean) for each atribute of each class (train data)
	set_msd, ver_msd, vir_msd, prob = atributeProb(train_data)

	for i in range(len(test_data)):
		# Calculates the estimation for the current test case
		ret = classProb(test_data[i], prob, set_msd, ver_msd, vir_msd)

		if ret ==  test_data_result[i]:
			count += 1

	return count / len(test_data)
	

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

	acc = classification(train_data, test_data, test_data_result)

	print(50 * '-')
	print('...:: Nayve Bayes Algorithm - IRIS DataSet ::...')
	print('\nTest Correctness = {:.2f}%'.format(acc * 100))
	print(50 * '-')


if __name__ == '__main__':
	main() 
