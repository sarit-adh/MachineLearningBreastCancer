from sklearn import tree
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cross_validation import cross_val_score

file = pd.read_csv("data.csv")
df = pd.DataFrame(file)



x = df.loc[:,'radius_mean':'fractal_dimension_worst']
y = df.loc[:,'diagnosis']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

clf = tree.DecisionTreeClassifier(criterion='gini')
clf.fit(X_train,y_train)
scores = cross_val_score(clf,X_train,y_train, cv=10 )
print(scores.mean(), scores.std())

from sklearn.metrics import accuracy_score

y_pred_tr = clf.predict(X_test)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_tr))

from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
feature_names = df.columns[:30]
dot_data = export_graphviz(clf,out_file=None, feature_names=feature_names, class_names=None, filled=True, rounded=True, special_characters=True )
graph = pydotplus.graph_from_dot_data(dot_data)
#graph.write_pdf("hello2.pdf")