# SVM example from https://www.kdnuggets.com/2017/02/yhat-support-vector-machine.html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model, svm, tree


def plot_results_with_hyperplane(clf, clf_name, df, plt_nmbr):
	x_min, x_max = df.x.min() - 0.5, df.x.max() + 0.5
	y_min, y_max = df.y.min() - 0.5, df.y.max() + 0.5
	step = 0.2

	xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	plt.subplot(2, 2, plt_nmbr)
	plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

	for animal in df.animal.unique():
		plt.scatter(df[df.animal == animal].x, df[df.animal == animal].y, marker=animal, label='cows' if animal == 'x' else 'wolves', color='black')
		plt.title(clf_name)
		plt.legend(loc='best')


if __name__ == '__main__':
	data = open('cows_and_wolves.txt').read()
	data = [row.split('\t') for row in data.strip().split('\n')]

	animals = []
	for y, row in enumerate(data):
		for x, item in enumerate(row):
			# anim is cow, o is wolf
			if item in ['x', 'o']:
				animals.append([x, y, item])

	df = pd.DataFrame(animals, columns=['x', 'y', 'animal'])
	df['animal_type'] = df.animal.apply(lambda anim: 0 if anim == 'x' else 1)  # cows are 0, wolves are 1

	# train on anim and y coordinates
	train_cols = ['x', 'y']

	clfs = {'SVM': svm.SVC(), 'Logistic': linear_model.LogisticRegression(), 'Decision Tree': tree.DecisionTreeClassifier()}

	plt_nmbr = 1
	for clf_name, clf in clfs.items():
		clf.fit(df[train_cols], df.animal_type)
		plot_results_with_hyperplane(clf, clf_name, df, plt_nmbr)
		plt_nmbr += 1
	plt.show()
