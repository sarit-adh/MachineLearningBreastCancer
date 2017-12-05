from util import *
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder


#read data from file
filepath = "./data.csv"
bc_data = pd.read_csv(filepath) 


#split the dataset into train and test set
bc_df_values = bc_data.values
X = bc_df_values[:,2:31]
Y = bc_df_values[:,1]
test_set_size = 0.2

le = LabelEncoder()
le.fit(Y)
Y = le.transform(Y)
print le.transform(["M", "M", "B"]) #malignant 1 and benign 0 
seed = 1
scoring = 'accuracy' #other possible values precision, recall, f_score
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size = test_set_size,random_state=seed) 



#print X_train #get a peek at training data

#hyperparameter (C) tuning using cross validation
kfold = model_selection.KFold(n_splits=10, random_state=seed) # try with LOOCV too
#Inverse of regularization strength; smaller values specify stronger regularization
C_values_list = [0.001, 0.01, 0.1, 1, 10,50,100,1000]

C_score_resuls_list =[]
C_score_dict = {}
for C_value in C_values_list:
	regression_model = LogisticRegression(C=C_value)
	cv_results = model_selection.cross_val_score(regression_model, X_train, Y_train, cv=kfold, scoring=scoring)

	C_score_dict[C_value] =  (cv_results.mean(),cv_results.std())
	C_score_resuls_list.append(cv_results)
	
print C_score_dict
C_best = max(C_score_dict, key=C_score_dict.get)
print "Best value for C is: " + str(C_best)

# Compare Hyperparameters
fig = plt.figure()
fig.suptitle('Hyperparameter Tuning for logistic regression')
ax = fig.add_subplot(111)
plt.boxplot(C_score_resuls_list)
ax.set_xticklabels(C_values_list)
ax.set_xlabel("regularization parameter")
ax.set_ylabel("accuracy")
plt.show()

print Y_test



#performance on test dataset
regression_model = LogisticRegression(C=C_best)
regression_model.fit(X_train,Y_train)
predictions = regression_model.predict(X_test)
#print predictions
print "Confusion Matrix"
print confusion_matrix(Y_test, predictions)

print "Accuracy"
print accuracy_score(Y_test, predictions)

print "Precision"
print precision_score(Y_test, predictions)

print "Recall"
print recall_score(Y_test, predictions)

print "f-score"
print f1_score(Y_test,predictions)

for i in range(0,len(predictions)):
	if predictions[i] != Y_test[i]:
		print i