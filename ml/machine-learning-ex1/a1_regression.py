import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model
from mpl_toolkits import mplot3d

def oneD_regression(file):
	data = np.genfromtxt(file, delimiter=',')
	#x = data[:,0] #this is (n,) shape. Screws up regression call
	x = data[:,0:1] #this is (n,1) shape
	y = data[:,1] #this is (n,1) shape
	plt.figure("2d plot")
	plt.scatter(x, y)
	plt.xlabel('population')
	plt.ylabel('profit')
	Lreg = linear_model.LinearRegression() #Calculates OLS analytical solution with L2 norm cost function. Assumes terms are independent
	Lreg.fit(x, y)
	plt.plot(x, Lreg.predict(x))
	plt.show()

def twoD_regression(file):
	data = np.genfromtxt(file, delimiter=',')
	x = data[:,0:2]
	y = data[:,2]
	plt.figure("3d plot")
	plt.gca(projection='3d').scatter(x[:,0], x[:,1], y)
	Lreg = linear_model.LinearRegression() #As before but is multivariate in this case. Solution is now a plane not a line (y = ax_0 + bx_1 + c)
	Lreg.fit(x, y)
	x0, x1 = np.meshgrid(x[:,0], x[:,1])
	plt.gca(projection='3d').plot_wireframe(x0, x1, Lreg.predict(x), color='r')
	plt.show()


def main():
	file1 = '/home/user/coursera_ML/machine-learning-ex1/ex1/ex1data1.txt'
	#oneD_regression(file=file1)
	file2 = '/home/user/coursera_ML/machine-learning-ex1/ex1/ex1data2.txt'
	
	twoD_regression(file=file2)

if __name__ == '__main__':
	main()