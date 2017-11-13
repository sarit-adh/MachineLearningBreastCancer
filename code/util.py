import pandas as pd
import os


#Function to read data from file
def read_data_from_file(filepath,column_names,index_column):
	df = pd.read_table(filepath,names=column_names,sep=',')
	df.set_index(index_column)
	return df

def print_summary(df):
	print "Total records: " + str(df.shape[0])
	print "Index column: " + str(df.index)
	print "Total Columns " + str(df.shape[1])
	print "Data Statistics"
	print df.iloc[:,2:len(df.columns)].describe()
	print "class distribution"
	print(df.groupby('Diagnosis').size())
	
#https://chrisalbon.com/python/pandas_indexing_selecting.html
def select_rows(df,start=0,end=0):
	
	if type(start)==int:
		return df.iloc[start:end,:]
	else:
		return df.loc[:start]
	
def select_columns(df,columns_list):
	return df.iloc[:,columns_list]
	
	
	
	



