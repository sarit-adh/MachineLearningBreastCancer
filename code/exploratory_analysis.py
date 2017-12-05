from util import *
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

#datafile to dataframe
# filepath = "../data/wdbc.data"
# column_names = ["ID_number", "Diagnosis", "radius_mean", "texture_mean","perimeter_mean","area_mean","smoothness_mean", \
				# "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean", \
				# "radius_SE", "texture_SE","perimeter_SE","area_SE","smoothness_SE", \
				# "compactness_SE","concavity_SE","concave_points_SE","symmetry_SE","fractal_dimension_SE", \
				# "radius_worst", "texture_worst","perimeter_worst","area_worst","smoothness_worst", \
				# "compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"]

# bc_data = read_data_from_file(filepath,column_names,"ID_number")

#read data from file
filepath = "./data.csv"
bc_data = pd.read_csv(filepath) 


#split the dataset into train and test set
bc_df_values = bc_data.values
X = bc_df_values[:,2:31]
Y = bc_df_values[:,1]

#exploratory data analysis
#print_summary(bc_data)
#bc_data_subset = select_columns(bc_data,[2,3,4,5])

#univariate plot
#bc_data_subset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

#multivariate plot
#scatter_matrix(bc_data)
#plt.show()

#sns.pairplot(bc_data, hue="Diagnosis")
#plt.show()

X_embedded = TSNE(n_components=2).fit_transform(X)
print X_embedded

df_tsne = pd.DataFrame(X_embedded, columns=['dim1','dim2'])
df_tsne['class'] = Y 



g = sns.lmplot('dim1', 'dim2', df_tsne, hue='class', fit_reg=False, size=8
                ,scatter_kws={'alpha':0.7,'s':60})
g.axes.flat[0].set_title('Scatterplot of breast cancer dataset reduced to 2D using t-SNE')

g.show()





