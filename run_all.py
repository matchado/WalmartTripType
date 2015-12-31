
import os, shutil

if __name__ == '__main__':
	processed_data_folder = os.path.join(os.getcwd(), 'processed_data')
	
	if os.path.exists(processed_data_folder):
		shutil.rmtree(processed_data_folder)
	
	os.makedirs(processed_data_folder)
		
	execfile('aggregate_features.py')
	execfile('extract_features.py')
	execfile('train_and_predict.py')