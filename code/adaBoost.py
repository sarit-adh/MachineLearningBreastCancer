from util import *
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


#read data from file
filepath = "./data.csv"
bc_data = pd.read_csv(filepath) 



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
#C_values_list = [0.001, 0.01, 0.1, 1, 10,50,100,1000]


decisionTreeClassifier = DecisionTreeClassifier()
adaBoostClassifier = AdaBoostClassifier(base_estimator = decisionTreeClassifier)
#X_train, X_val , y_train, y_val = train_test_split(X_train, y_train, test_size=0.1,random_state=seed)


param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
			  "learning_rate" : [0.01,0.1,0.05,0.2,0.04],
              "n_estimators": [1,2]
             }

			 
			 
gridSearchCV = GridSearchCV(adaBoostClassifier, param_grid=param_grid, scoring = scoring, cv=kfold)			 
gridSearchCV.fit(X_train,Y_train)
print gridSearchCV.grid_scores_

#C_score_resuls_list =[]
#C_score_dict = {}
#for C_value in C_values_list:
#	regression_model = LogisticRegression(C=C_value)
#	cv_results = model_selection.cross_val_score(regression_model, X_train, Y_train, cv=kfold, scoring=scoring)
#
#	C_score_dict[C_value] =  (cv_results.mean(),cv_results.std())
#	C_score_resuls_list.append(cv_results)
#	
#print C_score_dict
#C_best = max(C_score_dict, key=C_score_dict.get)
#print "Best value for C is: " + str(C_best)

# Compare Hyperparameters
#fig = plt.figure()
#fig.suptitle('Hyperparameter Comparison')
#ax = fig.add_subplot(111)
#plt.boxplot(C_score_resuls_list)
#ax.set_xticklabels(C_values_list)
#plt.show()

#performance on test dataset
#regression_model = LogisticRegression(C=C_best)
#regression_model.fit(X_train,Y_train)
predictions = gridSearchCV.predict(X_test)
print predictions
print confusion_matrix(Y_test, predictions)
print accuracy_score(Y_test, predictions)