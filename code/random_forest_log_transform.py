from util import *
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier

#read data from file
filepath = "./data.csv"
bc_data = pd.read_csv(filepath) 
print bc_data.head()

bc_data_id_target=bc_data.loc[:,["id","diagnosis"]]

#Log transform data in dataframe
#log_columns = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean',
#       'concave points_mean','symmetry_mean']

#bc_data[log_columns] = bc_data[log_columns].apply(np.log10)

X_log_transform = bc_data.iloc[:,2:].apply(np.log10)

bc_data = bc_data_id_target.join(X_log_transform)

print bc_data.head()



#split the dataset into train and test set
bc_df_values = bc_data.values
X = bc_df_values[:,2:31]
Y = bc_df_values[:,1]
test_set_size = 0.2

print np.any(np.isnan(X.astype(float)))
print np.any(np.isinf(X.astype(float)))
print np.argwhere(np.isinf(X.astype(float)))

max_float_value = sys.float_info.max
min_float_value = sys.float_info.min

X[X > max_float_value] = max_float_value
X[X < min_float_value] = min_float_value

#log transorm data
#X_log = np.log10(X)



seed = 1
scoring = 'accuracy' #other possible values precision, recall, f_score
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size = test_set_size,random_state=seed) 

#print X_train #get a peek at training data

#hyperparameter (C) tuning using cross validation
kfold = model_selection.KFold(n_splits=10, random_state=seed) # try with LOOCV too
#Number of estimators
N_values_list = [10,50,100,1000]

N_score_resuls_list =[]
N_score_dict = {}
for N_value in N_values_list:
	randomForestClassifier = RandomForestClassifier(n_estimators=N_value)
	cv_results = model_selection.cross_val_score(randomForestClassifier, X_train, Y_train, cv=kfold, scoring=scoring)

	N_score_dict[N_value] =  (cv_results.mean(),cv_results.std())
	N_score_resuls_list.append(cv_results)
	
print N_score_dict
N_best = max(N_score_dict, key=N_score_dict.get)
print "Best value for N is: " + str(N_best)

# Compare Hyperparameters
fig = plt.figure()
fig.suptitle('Hyperparameter Comparison')
ax = fig.add_subplot(111)
plt.boxplot(N_score_resuls_list)
ax.set_xticklabels(N_values_list)
plt.show()

#performance on test dataset
regression_model = RandomForestClassifier(n_estimators=N_best)
regression_model.fit(X_train,Y_train)
predictions = regression_model.predict(X_test)
print predictions
print confusion_matrix(Y_test, predictions)
print accuracy_score(Y_test, predictions)