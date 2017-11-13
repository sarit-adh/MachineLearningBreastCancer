from util import *
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection

#datafile to dataframe
filepath = "../data/wdbc.data"
column_names = ["ID_number", "Diagnosis", "radius_mean", "texture_mean","perimeter_mean","area_mean","smoothness_mean", \
				"compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean", \
				"radius_SE", "texture_SE","perimeter_SE","area_SE","smoothness_SE", \
				"compactness_SE","concavity_SE","concave_points_SE","symmetry_SE","fractal_dimension_SE", \
				"radius_worst", "texture_worst","perimeter_worst","area_worst","smoothness_worst", \
				"compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"]

bc_data = read_data_from_file(filepath,column_names,"ID_number")


#exploratory data analysis
#print_summary(bc_data)
#bc_data_subset = select_columns(bc_data,[2,3,4,5])

#univariate plot
#bc_data_subset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

#multivariate plot
#scatter_matrix(bc_data_subset)
#plt.show()

#split the dataset into train and test set
bc_df_values = bc_data.values
X = bc_df_values[:,2:31]
Y = bc_df_values[:,1]
test_set_size = 0.2
seed = 1
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size = test_set_size,random_state=seed) 

print X_train

#print select_rows(bc_data,"84300903")


#print select_rows(bc_data,0,3)
#print select_columns(bc_data,[1,3])

