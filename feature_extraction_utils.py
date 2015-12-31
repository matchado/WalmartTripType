import os, cPickle
import numpy as np

processed_data_folder = os.path.join(os.getcwd(), 'processed_data')

incon = open(os.path.join(processed_data_folder, 'dept_x_fine_line_agg_map_v4.pkl'), 'rU')
dept_x_fine_line_agg_map_v4 = cPickle.load(incon)
incon.close()

incon = open(os.path.join(processed_data_folder, 'dept_x_fine_line_agg_map_v6.pkl'), 'rU')
dept_x_fine_line_agg_map_v6 = cPickle.load(incon)
incon.close()

incon = open(os.path.join(processed_data_folder, 'dept_x_fine_line_agg_map_v7.pkl'), 'rU')
dept_x_fine_line_agg_map_v7 = cPickle.load(incon)
incon.close()

incon = open(os.path.join(processed_data_folder, 'fine_line_x_upc_agg_map_v5.pkl'), 'rU')
fine_line_x_upc_agg_map_v5 = cPickle.load(incon)
incon.close()


def my_tokenizer(line):
    return line.split("|")
	
def aggregate_data(name, df, which_data='train'):

    if which_data == 'train':
        trip_type = df.TripType.iloc[0]
    elif which_data == 'test':
        trip_type = 0
    else:
        print 'something wrong'
    
    grou = df.groupby('dept_cleaned')
    
    # max_prods_bought_in_any_dept
    max_prods_bought_in_any_dept = grou.actual_scan_count.sum().max()
    # eff_max_prods_bought_in_any_dept
    eff_max_prods_bought_in_any_dept = grou.ScanCount.sum().max()
    
    grou = df.groupby('fine_line_cleaned')

    # max_prods_bought_in_any_fine_line
    max_prods_bought_in_any_fine_line = grou.actual_scan_count.sum().max()
    # eff_max_prods_bought_in_any_fine_line
    eff_max_prods_bought_in_any_fine_line = grou.ScanCount.sum().max()
    
    
    # eff_total_prods_bought
    eff_total_prods_bought = df.ScanCount.sum()
    
    df_non_ret = df[df.ScanCount > -1]
    
    # nunique_depts_non_ret
    nunique_depts_non_ret = df_non_ret.dept_cleaned.nunique()
    # nunique_fine_lines_non_ret
    nunique_fine_lines_non_ret = df_non_ret.fine_line_cleaned.nunique()
    # nunique_upc_non_ret
    nunique_upc_non_ret = df_non_ret.upc_cleaned.nunique()
    
    
    total_prods_returned = (df.ScanCount < 0).sum()
    
    if total_prods_returned == 0:
        total_pieces_returned, max_pieces_returned = [0, 0]
        n_uniq_dept_returned, n_uniq_fine_line_returned = [0, 0]
        max_prods_returned_in_any_fine_line, max_prods_returned_in_any_dept = [0, 0]
    else:
        total_pieces_returned  = df.actual_scan_count[df.ScanCount < 0].sum()
        max_pieces_returned = df.actual_scan_count[df.ScanCount < 0].max()
        n_uniq_dept_returned = df.dept_cleaned[df.ScanCount < 0].nunique()
        n_uniq_fine_line_returned = df.fine_line_cleaned[df.ScanCount < 0].nunique()
        
        df_ret = df[df.ScanCount < 0]
        
        # max_prods_returned_in_any_fine_line
        max_prods_returned_in_any_fine_line = df_ret.groupby('fine_line_cleaned').actual_scan_count.sum().max()
        # max_prods_returned_in_any_dept
        max_prods_returned_in_any_dept = df_ret.groupby('dept_cleaned').actual_scan_count.sum().max()
    
    return [name,
        #TripType, nunique_depts, nunique_upc,
        trip_type, df.dept_cleaned.nunique(), df.upc_cleaned.nunique(), 
        # num_of_unknown_products, nunique_fine_line, num_of_unknown_fineline,
        (df.upc_cleaned == 'unknown_product').sum(), df.fine_line_cleaned.nunique(), (df.fine_line_cleaned == 'unknown_fineline').sum(),
        # total_prods, total_pieces_bought, max_pieces_bought,
        len(df), df.actual_scan_count.sum(), df.actual_scan_count.max(),
        total_prods_returned, total_pieces_returned, max_pieces_returned,
        n_uniq_dept_returned, n_uniq_fine_line_returned,
        max_prods_bought_in_any_dept, eff_max_prods_bought_in_any_dept,
        max_prods_bought_in_any_fine_line, eff_max_prods_bought_in_any_fine_line,
        eff_total_prods_bought, nunique_depts_non_ret,
        nunique_fine_lines_non_ret, nunique_upc_non_ret,
        max_prods_returned_in_any_fine_line, max_prods_returned_in_any_dept]
	

def extract_feature_v2(df):
    grou = df.groupby(['upc_cleaned','dept_cleaned','fine_line_cleaned']).ScanCount.sum().reset_index()
    return "|".join(["|".join(["_".join(str(x).split())]*y) for x, y in zip(grou.dept_cleaned, grou.ScanCount)])
	
def extract_feature_v3(df):
    grou = df[df.ScanCount > 0]
    return "|".join(["|".join(["_".join(str(x).split())]*y) for x, y in zip(grou.dept_cleaned, grou.ScanCount)])
	
def extract_feature_v4(df):
	grou = df[df.ScanCount > 0]
	return "|".join(["|".join(['dept_x_fine_line_'+str(dept_x_fine_line_agg_map_v4.get("_".join(x.split()).lower() + str(y),'un_mapped'))]*z) for x, y, z in zip(grou.dept_cleaned, grou.fine_line_cleaned, grou.ScanCount)])
	
def extract_feature_v6(df):
    grou = df.groupby(['upc_cleaned','dept_cleaned','fine_line_cleaned']).ScanCount.sum().reset_index()
    grou = grou[grou.ScanCount > 0]
    return "|".join(["|".join(['dept_x_fine_line_agg_'+str(dept_x_fine_line_agg_map_v6.get("_".join(x.split()).lower() + str(y),'un_mapped'))]*z) for x, y, z in zip(grou.dept_cleaned, grou.fine_line_cleaned, grou.ScanCount)])
	
def extract_feature_v7(df):
    grou = df.groupby(['upc_cleaned','dept_cleaned','fine_line_cleaned']).ScanCount.sum().reset_index()
    grou = grou[grou.ScanCount > 0]
    return "|".join(["|".join(['dept_x_fine_line_agg_'+str(dept_x_fine_line_agg_map_v7.get("_".join(x.split()).lower() + str(y),'un_mapped'))]*z) for x, y, z in zip(grou.dept_cleaned, grou.fine_line_cleaned, grou.ScanCount)])

def extract_feature_v5(df):
    grou = df.groupby(['upc_cleaned','dept_cleaned','fine_line_cleaned']).ScanCount.sum().reset_index()
    grou = grou[grou.ScanCount > 0]
    return "|".join(["|".join(['fine_line_x_upc_agg_'+str(fine_line_x_upc_agg_map_v5.get(str(x) + "_" + str(y),'un_mapped'))]*z) for x, y, z in zip(grou.fine_line_cleaned, grou.upc_cleaned, grou.ScanCount)])
	
def extract_return_feature(df):
    te = df[df.ScanCount < 0]
    if len(te) > 0:
        return "|".join(["|".join(["_".join(str(x).split())]*y) for x, y in zip(te.dept_cleaned, te.actual_scan_count)])
    else:
        return ''

