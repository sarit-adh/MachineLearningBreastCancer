from util import *
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt

filepath = "../data/wdbc.data"
column_names = ["ID_number", "Diagnosis", "radius_mean", "texture_mean","perimeter_mean","area_mean","smoothness_mean", \
				"compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean", \
				"radius_SE", "texture_SE","perimeter_SE","area_SE","smoothness_SE", \
				"compactness_SE","concavity_SE","concave_points_SE","symmetry_SE","fractal_dimension_SE", \
				"radius_worst", "texture_worst","perimeter_worst","area_worst","smoothness_worst", \
				"compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"]

bc_data = read_data_from_file(filepath,column_names,"ID_number")


#split the dataset into train and test set
bc_df_values = bc_data.values
X = bc_df_values[:,2:31]
Y = bc_df_values[:,1]
test_set_size = 0.2

seed = 1
scoring = 'accuracy' #other possible values precision, recall, f_score


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size = test_set_size,random_state=seed) 



#print X_train #get a peek at training data

#hyperparameter (C) tuning using cross validation
kfold = model_selection.KFold(n_splits=10, random_state=seed) # try with LOOCV too
#Inverse of regularization strength; smaller values specify stronger regularization
C_values_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

C_score_resuls_list =[]
C_score_dict = {}
for C_value in C_values_list:
	regression_model = LogisticRegression(C=C_value)
	cv_results = model_selection.cross_val_score(regression_model, X_train, Y_train, cv=kfold, scoring=scoring)

	C_score_dict[C_value] =  cv_results.mean()
	C_score_resuls_list.append(cv_results)
	
print C_score_dict
print "Best value for C is: " + str(max(C_score_dict, key=C_score_dict.get))

# Compare Hyperparameters
fig = plt.figure()
fig.suptitle('Hyperparameter Comparison')
ax = fig.add_subplot(111)
plt.boxplot(C_score_resuls_list)
ax.set_xticklabels(C_values_list)
plt.show()