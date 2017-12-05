import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from sklearn.metrics import log_loss,accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

#Read data from csv
breast_cancer_ds = pd.read_csv("data.csv")

#Find number of records and features
num_records = breast_cancer_ds.shape[0]
num_features = breast_cancer_ds.shape[1] - 2


X = breast_cancer_ds.iloc[:,2:num_features+1]
y = breast_cancer_ds['diagnosis']

#label encode output
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

print y[0:25]

test_set_size = 0.2
seed = 1


#split into train set and test set

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size = test_set_size,random_state=seed)

classifiers = []

X_train, X_val , y_train, y_val = train_test_split(X_train, y_train, test_size=0.1,random_state=1)

#logisticRegressionClassifier = LogisticRegression(random_state=1,C=1000)
#logisticRegressionClassifier.fit(X_train,y_train)
#print('Logistic Regression Classifier LogLoss {score}').format(score=log_loss(y_val,logisticRegressionClassifier.predict_proba(X_val)))
#classifiers.append(logisticRegressionClassifier)

# logisticRegressionClassifier2 = LogisticRegression(random_state=2,C=1000)
# logisticRegressionClassifier2.fit(X_train,y_train)
# print('Logistic Regression Classifier LogLoss {score}').format(score=log_loss(y_val,logisticRegressionClassifier2.predict_proba(X_val)))
# classifiers.append(logisticRegressionClassifier2)

gradientBoostingClassifier = GradientBoostingClassifier()
gradientBoostingClassifier.fit(X_train,y_train)
print('Gradient Boosting Classifier LogLoss {score}').format(score=log_loss(y_val,gradientBoostingClassifier.predict_proba(X_val)))
classifiers.append(gradientBoostingClassifier)

#decisionTreeClassifier = DecisionTreeClassifier(criterion='entropy', max_depth=10, max_leaf_nodes= 18, min_samples_leaf=4, min_samples_split=8)
#decisionTreeClassifier.fit(X_train,y_train) 
#print('Decision Tree Classifier LogLoss {score}').format(score=log_loss(y_val,decisionTreeClassifier.predict_proba(X_val)))
#classifiers.append(decisionTreeClassifier)

### finding the optimal weights
predictions =[]
for classifier in classifiers:
        predictions.append(classifier.predict_proba(X_val))

def log_loss_func(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        final_prediction = 0
        for weight,prediction in zip(weights,predictions):
                final_prediction += weight*prediction
        return log_loss(y_val, final_prediction)
starting_values = [0.5]* len(predictions)

constraints_dict = ({'type' : 'eq' , 'fun' : lambda w: 1-sum(w)})
bounds = [(0,1)] * len(predictions)

res = minimize(log_loss_func, starting_values, method='SLSQP' , bounds=bounds, constraints = constraints_dict)

print('Ensemble Score: {best_score}').format(best_score=res['fun'])
print('Best Weights: {weights}').format(weights=res['x'])

def make_prediction(weights,classifiers,test_features):
	prediction =0 
	for weight,classifier in zip(weights,classifiers):
		prediction += weight * classifier.predict_proba(test_features)
	return prediction

prediction = make_prediction(res['x'],classifiers,X_test)


prediction = [elem[0]<0.4 for elem in prediction]

print len(y_test)
#print prediction
print accuracy_score(y_test,prediction)