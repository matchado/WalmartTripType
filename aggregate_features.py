import os, cPickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def aggregate_feature_v6(df):
    grou = df.groupby(['upc_cleaned','dept_cleaned','fine_line_cleaned']).ScanCount.sum().reset_index()
    grou = grou[grou.ScanCount > 0]
    return [("_".join(x.split()).lower() + str(y), df.TripType.iloc[0], z) for x, y, z in zip(grou.dept_cleaned, grou.fine_line_cleaned, grou.ScanCount)]
	
def aggregate_feature_v5(df):
    grou = df.groupby(['upc_cleaned','dept_cleaned','fine_line_cleaned']).ScanCount.sum().reset_index()
    grou = grou[grou.ScanCount > 0]
    return [(str(x) + "_" + str(y), df.TripType.iloc[0], z) for x, y, z in zip(grou.fine_line_cleaned, grou.upc_cleaned, grou.ScanCount)]

def create_aggregation_map(df, min_occurence, agg_name):
	counts = df.apply(lambda x: x.sum(), 1)
	dag1 = df[counts > min_occurence]
	dag2 = dag1.apply(lambda x: x/x.sum(), 1)
	percs = np.array(dag2)

	km = KMeans(200, random_state = 1234)
	clust_labels = km.fit_predict(percs)
	agg_map = {dag2.index[i] : clust_labels[i] for i in range(len(clust_labels))}
	
	output_file_name = os.path.join(os.getcwd(), 'processed_data', agg_name+'.pkl')
	outcon = open(output_file_name, 'wb')
	cPickle.dump(agg_map, outcon)
	outcon.close()


data = pd.read_csv(os.path.join(os.getcwd(), 'train.csv'))
data['upc_cleaned'] = data.Upc.fillna('unknown_product')
data['dept_cleaned'] = data.DepartmentDescription.fillna('unknown_dept')
data['fine_line_cleaned'] = data.FinelineNumber.fillna('unknown_fineline')
data['actual_scan_count'] = data.ScanCount.abs()
grouped = data.groupby('VisitNumber')
	
# aggregate dept * fine line features
data1 = data[data.ScanCount > 0]
data1['dept_x_fine_line'] =  ["_".join(x.split()).lower() + str(y) for x, y in zip(data1.dept_cleaned, data1.fine_line_cleaned)]
grouped = data1.groupby(['dept_x_fine_line','TripType'])
da1 = grouped.ScanCount.sum().reset_index()
dag = da1.pivot(index = 'dept_x_fine_line', columns='TripType', values = 'ScanCount').fillna(0)

create_aggregation_map(dag, 10, 'dept_x_fine_line_agg_map_v4')


# aggregate dept * fine line features after removing the returned products
temp_holder1 = [aggregate_feature_v6(df) for name, df in grouped]
temp_holder = [item for sublist in temp_holder1 for item in sublist]
da1 = pd.DataFrame(temp_holder)
da1.columns = ['dept_x_fine_line','TripType','actual_scan_count']
grouped1 = da1.groupby(['dept_x_fine_line','TripType'])
da2 = grouped1.actual_scan_count.sum().reset_index()
dag = da2.pivot(index = 'dept_x_fine_line', columns='TripType', values = 'actual_scan_count').fillna(0)

create_aggregation_map(dag, 10, 'dept_x_fine_line_agg_map_v6')
create_aggregation_map(dag, 20, 'dept_x_fine_line_agg_map_v7')


# aggregate fine line * upc features after removing the returned products
temp_holder1 = [aggregate_feature_v5(df) for name, df in grouped]
temp_holder = [item for sublist in temp_holder1 for item in sublist]
da1 = pd.DataFrame(temp_holder)
da1.columns = ['fine_line_x_upc','TripType','actual_scan_count']
grouped1 = da1.groupby(['fine_line_x_upc','TripType'])
da2 = grouped1.actual_scan_count.sum().reset_index()
dag = da2.pivot(index = 'fine_line_x_upc', columns='TripType', values = 'actual_scan_count').fillna(0)

create_aggregation_map(dag, 20, 'fine_line_x_upc_agg_map_v5')
print "Finished creating feature aggregation maps"