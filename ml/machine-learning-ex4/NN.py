import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model, neural_network, model_selection, pipeline
from scipy import io

def bk_prop_NN(file):
	mat = io.loadmat(file)
	x = mat['X']
	y = mat['y']
	y = y.ravel()
	x_train, x_v, y_train, y_v = model_selection.train_test_split(x, y, test_size = 0.4)
	x_v, x_test, y_v, y_test = model_selection.train_test_split(x_v, y_v, test_size = 0.5)#splits data into 3 sets: training, validation and test, which hold 60%, 20% and 20% of the total data respectively.
	scale = preprocessing.StandardScaler()
	scale.fit(x_train) #gets fit for scaling transformation. N.b. just perform scaling on training data, as one should treat scaling as part of modelling process (i.e. train scaling on training set)
	xs_train = scale.transform(x_train) #performs scaling transformation
	mean, var = scale.mean_, scale.var_ #array of mean and variances for each feature
	NN = neural_network.MLPClassifier(hidden_layer_sizes=(15,15,15), alpha = 0.1) #NN with three hidden layers, each with 15 nodes each 
	NN.fit(xs_train, y_train)
	print NN.score(scale.transform(x_v), y_v)
	#ALTERNATIVELY, do k-fold cross validation, which only requires splitting into two sets of data. This is more computationally expensive, but better when you have limited training data
	x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.4)
	NN = pipeline.make_pipeline(preprocessing.StandardScaler(), neural_network.MLPClassifier(hidden_layer_sizes=(15,15,15), alpha = 0.1)) #make a pipe, which means that when cross validation is done, each training set is scaled and then trained from. Presumably when the NN is applied to the cross validation set this is also scaled by the same transformation derived from the training data
	scores = cross_val_score(NN, x_train, y_train, cv=5) #5-fold cross validation. I believe it shuffles the data for each iteration k (so folds don't have same data each time)
	scores.mean() #mean of score across the k iterations


def main():
	file1 = '/home/user/coursera_ML/machine-learning-ex4/ex4/ex4data1.mat'
	bk_prop_NN(file=file1)

if __name__ == '__main__':
	main()