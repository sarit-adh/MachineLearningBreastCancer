from util import *
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier

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

#hyperparameter (K) tuning using cross validation
kfold = model_selection.KFold(n_splits=10, random_state=seed) # try with LOOCV too
#Inverse of regularization strength; smaller values specify stronger regularization
K_values_list = [1,3, 5,7]

K_score_resuls_list =[]
K_score_dict = {}
for K_value in K_values_list:
	kNeighborsClassifier = KNeighborsClassifier(n_neighbors=K_value)
	cv_results = model_selection.cross_val_score(kNeighborsClassifier, X_train, Y_train, cv=kfold, scoring=scoring)

	K_score_dict[K_value] =  (cv_results.mean(),cv_results.std())
	K_score_resuls_list.append(cv_results)
	
print K_score_dict
K_best = max(K_score_dict, key=K_score_dict.get)
print "Best value for K is: " + str(K_best)

# Compare Hyperparameters
fig = plt.figure()
fig.suptitle('Hyperparameter Comparison')
ax = fig.add_subplot(111)
plt.boxplot(K_score_resuls_list)
ax.set_xticklabels(K_values_list)
plt.show()

#performance on test dataset
kNeighborsClassifier = KNeighborsClassifier(n_neighbors=K_best)
kNeighborsClassifier.fit(X_train,Y_train)
predictions = kNeighborsClassifier.predict(X_test)
print predictions
print confusion_matrix(Y_test, predictions)
print accuracy_score(Y_test, predictions)