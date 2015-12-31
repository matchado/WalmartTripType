import os, cPickle
import lasagne
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from scipy import sparse
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.updates import adagrad
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet
import xgboost as xgb


class TrainValidSplitter(object):
    def __init__(self, standardize=True, few=False):
        self.standardize = standardize
        self.few = few
        self.standa = None

    def __call__(self, X, y, net):
        strati = StratifiedShuffleSplit(y = y, n_iter = 1, test_size = 0.2, random_state = 1234)
        
        train_indices, valid_indices = next(iter(strati))
        
        if self.standardize:
            self.standa = StandardScaler()
            if self.few:
                X_train = np.hstack((self.standa.fit_transform(X[train_indices,:23]), X[train_indices,23:]))
                X_valid = np.hstack((self.standa.transform(X[valid_indices,:23]), X[valid_indices,23:]))
            else:
                X_train = self.standa.fit_transform(X[train_indices])
                X_valid = self.standa.transform(X[valid_indices])
        else:
            X_train, X_valid = X[train_indices], X[valid_indices]
        
        y_train, y_valid = y[train_indices], y[valid_indices]
        
        return X_train, X_valid, y_train, y_valid

class BestWeightsHolder(object):
	"""
	A class to hold the weights of the NeuralNet
	when the validation loss was at least seen
	"""
	def __init__(self):
		self.best_weights = []

	def hold_best_weights(self, nn, train_hist):
		# self.best_weights.append(nn.get_all_params_values())
		if train_hist[-1]['valid_loss_best']:
			self.best_weights = nn.get_all_params_values()

def fit_nn_and_predict_probas(features, dv, features_t):
	bwh = BestWeightsHolder()
	tvs = TrainValidSplitter(standardize=True,few=True)

	layers = [('input', InputLayer),
		   ('dense0', DenseLayer),
		   ('dropout0', DropoutLayer),
		   ('dense1', DenseLayer),
		   ('dropout1', DropoutLayer),
		   ('output', DenseLayer)]

	net = NeuralNet(layers=layers,
			input_shape=(None, features.shape[1]),
			dense0_num_units=512,
			dropout0_p=0.4,
			dense1_num_units=256,
			dropout1_p=0.4,
			output_num_units=38,
			output_nonlinearity=softmax,
			update=adagrad,
			update_learning_rate=0.02,
			train_split=tvs,
			verbose=1,
			max_epochs=40,
			on_epoch_finished=[bwh.hold_best_weights])

	holder = net.fit(features, dv)
	holder.load_params_from(bwh.best_weights)
	return holder.predict_proba(np.hstack((tvs.standa.transform(features_t[:,:23]), features_t[:,23:])))
	

np.random.seed(1234)
processed_data_folder = os.path.join(os.getcwd(), 'processed_data')

incon = open(os.path.join(processed_data_folder, 'train_features.pkl'), 'rU')
fts_6, x_tif_dept_v2, x_tif_dept_v3, x_tif_dept_x_fine_v4, x_tif_dept_x_fine_v6, x_tif_dept_x_fine_v7, x_tif_fine_x_upc_v5, x_tif_return_dept = cPickle.load(incon)
incon.close()

incon = open(os.path.join(processed_data_folder, 'test_features.pkl'), 'rU')
fts_6_t, x_tif_dept_v2_t, x_tif_dept_v3_t, x_tif_dept_x_fine_v4_t, x_tif_dept_x_fine_v6_t, x_tif_dept_x_fine_v7_t, x_tif_fine_x_upc_v5_t, x_tif_return_dept_t = cPickle.load(incon)
incon.close()

## extract the unique trip types 
## and recode them from 0 to 37
class_nums = sorted(list(fts_6.iloc[:,1].unique()))
class_mapping = {class_nums[i]: i for i in range(len(class_nums))}
transformed_dv = np.array([class_mapping[k] for k in fts_6.iloc[:,1]])


## create features for nn & xgb models
features_nn_1 = np.hstack((fts_6.iloc[:,2:], 
						sparse.hstack((x_tif_dept_v3,
										x_tif_dept_x_fine_v4)).toarray()))

features_nn_2 = np.hstack((fts_6.iloc[:,2:], 
						sparse.hstack((x_tif_dept_v2,
										x_tif_dept_x_fine_v6)).toarray()))

features_nn_3 = np.hstack((fts_6.iloc[:,2:], 
						sparse.hstack((x_tif_dept_v2,
										x_tif_dept_x_fine_v7)).toarray()))
										
features_xgb = sparse.hstack((fts_6.iloc[:,2:],
								x_tif_dept_v2,
								x_tif_dept_v3,
								x_tif_dept_x_fine_v6,
								x_tif_dept_x_fine_v7,
								x_tif_fine_x_upc_v5,
								x_tif_return_dept), 'csr')

## test features
features_nn_1_t = np.hstack((fts_6_t.iloc[:,2:], 
						sparse.hstack((x_tif_dept_v3_t,
										x_tif_dept_x_fine_v4_t)).toarray()))

features_nn_2_t = np.hstack((fts_6_t.iloc[:,2:], 
						sparse.hstack((x_tif_dept_v2_t,
										x_tif_dept_x_fine_v6_t)).toarray()))

features_nn_3_t = np.hstack((fts_6_t.iloc[:,2:], 
						sparse.hstack((x_tif_dept_v2_t,
										x_tif_dept_x_fine_v7_t)).toarray()))
										
features_xgb_t = sparse.hstack((fts_6_t.iloc[:,2:],
								x_tif_dept_v2_t,
								x_tif_dept_v3_t,
								x_tif_dept_x_fine_v6_t,
								x_tif_dept_x_fine_v7_t,
								x_tif_fine_x_upc_v5_t,
								x_tif_return_dept_t), 'csr')

predicted_probas = []

## build nn models
predicted_probas.append(fit_nn_and_predict_probas(features_nn_1, transformed_dv, features_nn_1_t))
predicted_probas.append(fit_nn_and_predict_probas(features_nn_2, transformed_dv, features_nn_2_t))
predicted_probas.append(fit_nn_and_predict_probas(features_nn_3, transformed_dv, features_nn_3_t))

## build xgb models

## training and validation data for building xgb models
strati = StratifiedShuffleSplit(y = transformed_dv, n_iter = 1, test_size = 0.2, random_state = 1234)
train_indices, valid_indices = next(iter(strati))
X_train, y_train = features_xgb[train_indices], transformed_dv[train_indices]
X_valid, y_valid = features_xgb[valid_indices], transformed_dv[valid_indices]

## xgb model 1
gbm = xgb.XGBClassifier(max_depth=8, n_estimators=1000, learning_rate=0.1, silent=False, 
                        colsample_bytree=0.5, min_child_weight=5, subsample=0.9)
gbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='mlogloss', early_stopping_rounds=5)

predicted_probas.append(gbm.predict_proba(features_xgb_t))


## xgb model 2
gbm = xgb.XGBClassifier(max_depth=8, n_estimators=1000, learning_rate=0.1, silent=False, 
                        colsample_bytree=0.5, min_child_weight=10, subsample=0.9)
gbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='mlogloss', early_stopping_rounds=5)

predicted_probas.append(gbm.predict_proba(features_xgb_t))

## ensemble the predicted probabilities from all the 5 models
total_probas = 0
for nparr in predicted_probas:
    total_probas = total_probas + nparr

ensemble_probas = np.apply_along_axis(lambda x: x/x.sum(), 1, total_probas)


output = pd.DataFrame(ensemble_probas)
output.columns = ['TripType_' + str(i) for i in class_nums]
output.insert(0, 'VisitNumber', fts_6_t.iloc[:,0])

output.to_csv(os.path.join(os.getcwd(),'ensembled_predicted_probabilities.csv'), index = False)