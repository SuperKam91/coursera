import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model
from scipy import io

def multi_class_LR(file):
	mat = io.loadmat(file)
	x = mat['X']
	y = mat['y']
	y = y.ravel()
	log_reg = linear_model.LogisticRegression() #logistic regression for multiple classes which uses one vs all system. This means it trains a classifier for each class (i.e. a record belongs to this class or it doesn't). The label is then determined by seeing which classifier gives the highest probability for the corresponding record
	log_reg.fit(x, y)
	print log_reg.score(x, y)

def main():
	file1 = '/home/user/coursera_ML/machine-learning-ex3/ex3/ex3data1.mat'
	multi_class_LR(file=file1)

if __name__ == '__main__':
	main()