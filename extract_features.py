import os, cPickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from feature_extraction_utils import *

## folder to store processed data
processed_data_folder = os.path.join(os.getcwd(), 'processed_data')

train_data = pd.read_csv(os.path.join(os.getcwd(),'train.csv'))
test_data = pd.read_csv(os.path.join(os.getcwd(),'test.csv'))

## impute missing values
train_data['upc_cleaned'] = train_data.Upc.fillna('unknown_product')
train_data['dept_cleaned'] = train_data.DepartmentDescription.fillna('unknown_dept')
train_data['fine_line_cleaned'] = train_data.FinelineNumber.fillna('unknown_fineline')
train_data['actual_scan_count'] = train_data.ScanCount.abs()

test_data['upc_cleaned'] = test_data.Upc.fillna('unknown_product')
test_data['dept_cleaned'] = test_data.DepartmentDescription.fillna('unknown_dept')
test_data['fine_line_cleaned'] = test_data.FinelineNumber.fillna('unknown_fineline')
test_data['actual_scan_count'] = test_data.ScanCount.abs()

# group by VisitNumber
grouped = train_data.groupby('VisitNumber')
grouped_t = test_data.groupby('VisitNumber')


# extract base features
temp_holder = [aggregate_data(name, df) for name, df in grouped]
fts_6 = pd.DataFrame(temp_holder)

temp_holder = [aggregate_data(name, df, 'test') for name, df in grouped_t]
fts_6_t = pd.DataFrame(temp_holder)

print "Finished extracting base features"

# extract dept_features v2
temp_holder = [extract_feature_v2(df) for name, df in grouped]
tif_dept_v2 = TfidfVectorizer(tokenizer = my_tokenizer, smooth_idf = False)
x_tif_dept_v2 = tif_dept_v2.fit_transform(temp_holder).tocsc()

temp_holder = [extract_feature_v2(df) for name, df in grouped_t]
x_tif_dept_v2_t = tif_dept_v2.transform(temp_holder).tocsc()

# extract dept_features v3
temp_holder = [extract_feature_v3(df) for name, df in grouped]
tif_dept_v3 = TfidfVectorizer(tokenizer = my_tokenizer, smooth_idf = False)
x_tif_dept_v3 = tif_dept_v3.fit_transform(temp_holder).tocsc()

temp_holder = [extract_feature_v3(df) for name, df in grouped_t]
x_tif_dept_v3_t = tif_dept_v3.transform(temp_holder).tocsc()

print "Finished extracting department features"

# extract dept * fine line features v4
temp_holder = [extract_feature_v4(df) for name, df in grouped]
tif_dept_x_fine_v4 = TfidfVectorizer(tokenizer = my_tokenizer, smooth_idf = False)
x_tif_dept_x_fine_v4 = tif_dept_x_fine_v4.fit_transform(temp_holder).tocsc()

temp_holder = [extract_feature_v4(df) for name, df in grouped_t]
x_tif_dept_x_fine_v4_t = tif_dept_x_fine_v4.transform(temp_holder).tocsc()

# extract dept * fine line features v6
temp_holder = [extract_feature_v6(df) for name, df in grouped]
tif_dept_x_fine_v6 = TfidfVectorizer(tokenizer = my_tokenizer)
x_tif_dept_x_fine_v6 = tif_dept_x_fine_v6.fit_transform(temp_holder).tocsc()

temp_holder = [extract_feature_v6(df) for name, df in grouped_t]
x_tif_dept_x_fine_v6_t = tif_dept_x_fine_v6.transform(temp_holder).tocsc()

# extract dept * fine line features v7
temp_holder = [extract_feature_v7(df) for name, df in grouped]
tif_dept_x_fine_v7 = TfidfVectorizer(tokenizer = my_tokenizer)
x_tif_dept_x_fine_v7 = tif_dept_x_fine_v7.fit_transform(temp_holder).tocsc()

temp_holder = [extract_feature_v7(df) for name, df in grouped_t]
x_tif_dept_x_fine_v7_t = tif_dept_x_fine_v7.transform(temp_holder).tocsc()

print "Finished extracting department * fine line features"

# extract fine line * upc features v5
temp_holder = [extract_feature_v5(df) for name, df in grouped]
tif_fine_x_upc_v5 = TfidfVectorizer(tokenizer = my_tokenizer)
x_tif_fine_x_upc_v5 = tif_fine_x_upc_v5.fit_transform(temp_holder).tocsc()

temp_holder = [extract_feature_v5(df) for name, df in grouped_t]
x_tif_fine_x_upc_v5_t = tif_fine_x_upc_v5.transform(temp_holder).tocsc()

print "Finished extracting fine line * upc features"


# extract return department features
temp_holder = [extract_return_feature(df) for name, df in grouped]
tif_return_dept = TfidfVectorizer(tokenizer = my_tokenizer, min_df = 30)
x_tif_return_dept = tif_return_dept.fit_transform(temp_holder).tocsc()

temp_holder = [extract_return_feature(df) for name, df in grouped_t]
x_tif_return_dept_t = tif_return_dept.transform(temp_holder).tocsc()

print "Finished extracting return department features"

print "saving the extracted train and test features to disk"

outcon = open(os.path.join(processed_data_folder, 'train_features.pkl'), 'wb')
cPickle.dump([fts_6, x_tif_dept_v2, x_tif_dept_v3,
				x_tif_dept_x_fine_v4, x_tif_dept_x_fine_v6, x_tif_dept_x_fine_v7, 
				x_tif_fine_x_upc_v5, x_tif_return_dept], outcon)
outcon.close()

outcon = open(os.path.join(processed_data_folder, 'test_features.pkl'), 'wb')
cPickle.dump([fts_6_t, x_tif_dept_v2_t, x_tif_dept_v3_t,
				x_tif_dept_x_fine_v4_t, x_tif_dept_x_fine_v6_t, x_tif_dept_x_fine_v7_t, 
				x_tif_fine_x_upc_v5_t, x_tif_return_dept_t], outcon)
outcon.close()

print "Done extracting features !!"