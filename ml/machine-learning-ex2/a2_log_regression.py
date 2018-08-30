import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model

def log_regres(file):
	data = np.genfromtxt(file, delimiter=',')
	x = data[:,0:2]
	y = data[:,2]
	m = ['o', 'x']
	plt.figure("2d scatter")
	plt.xlabel("test 1")
	plt.ylabel("test 2")
	log_reg = linear_model.LogisticRegression() #logistic regression which uses sigmoid logistic function to model probabilities. It uses a coordinate descent algorithm to find cost function minimum, and penalty term uses L2 norm. Regularisation term is inverse 
	log_reg.fit(x, y)
	#determine decision boundary
	lx_0, lx_1 = np.meshgrid(range(101), range(101)) #both unscaled features have a domain of [0, 100]
	ly = log_reg.predict(np.c_[lx_0.ravel(), lx_1.ravel()]) #gives 0 or 1 classification
	ly = ly.reshape(lx_0.shape)
	#plot decision boundary
	plt.contourf(lx_0, lx_1, ly)
	#plot original data with labels given by markers
	for i in range(len(y)):
		plt.scatter(x[i,0], x[i,1], marker = m[int(y[i])], color='k')
	#plt.show()
	#print log_reg.score(x, y)

def map_log_regres(file):
	data = np.genfromtxt(file, delimiter=',')
	x = data[:,0:2]
	y = data[:,2]
	m = ['o', 'x']
	plt.figure("2d scatter")
	plt.xlabel("test 1")
	plt.ylabel("test 2")
	#perform feature mapping
	feat_map = preprocessing.PolynomialFeatures(10) #consider up to 10th order polynomials
	map_x = feat_map.fit_transform(x)[:,:] 
	log_reg = linear_model.LogisticRegression() 
	log_reg.fit(map_x, y)
	lx = np.linspace(-1, 1.5, 50) #get array of points for domain of original features (both have ~ same domain)
	lx_0, lx_1 = np.meshgrid(lx, lx) #create grid of points for domain of original features
	ly = np.zeros((50,50)) #create zero grid for predictor over original feature domain
	for i in range(len(lx)):
		for j in range(len(lx)):
			ly[i,j] = log_reg.predict(feat_map.transform([[lx[i], lx[j]]])[:,:]) #loop over grid of orig feat values and calculate mapped feature values at corresponding points. Use these values to predict y over this same grid
	plt.contourf(lx_0, lx_1, ly) #plot contour of where y changes value over grid in original feature domain
	for i in range(len(y)):
		plt.scatter(x[i,0], x[i,1], marker = m[int(y[i])], color='k') #plot original data points
	plt.show()
	print log_reg.score(map_x, y)
		
def main():
	file1 = '/home/user/coursera_ML/machine-learning-ex2/ex2/ex2data1.txt'
	file2 = '/home/user/coursera_ML/machine-learning-ex2/ex2/ex2data2.txt'
	#log_regres(file=file1)
	map_log_regres(file=file2)

if __name__ == '__main__':
	main()