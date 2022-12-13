import os
import datetime
from jax import random
import sys
import numpy as onp
import jax.numpy as np
from jax.api import jit
from jax.config import config
from neural_tangents import stax
import neural_tangents as nt
from util import PredictionStatistics


class ActiveLearner(object):
	def __init__(self, args):
		self.args = args
		self.budget = args.budget
		self.active_iters = args.active_iters
		self.kernel_type = args.kernel_type
		self.biased_sample = args.biased_sample
		self.pred_stat = PredictionStatistics()

	def train(self, kernel_fn, X_train, Y_train, X_test = None, Y_test = None):
		kernel_fn = nt.batch(kernel_fn,
							 device_count=0,
							 batch_size=0)
		predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X_train,
															  Y_train, diag_reg=1e-3)
		if X_test is not None and Y_test is not None:
			self.test(predict_fn=predict_fn, X_test=X_test, Y_test=Y_test)
		return predict_fn

	def test(self, predict_fn, X_val, Y_val, query_infos_val, kernel_type="nngp", compute_cov = True):

		pred_mean, pred_cov = predict_fn(x_test=X_val, get=kernel_type,
									 compute_cov= compute_cov)
		errors = pred_mean - Y_val
		mse = np.mean(np.power(errors, 2.0))
		print("Test MSE Loss:{}".format(mse))
		self.pred_stat.get_prediction_details(errors, query_infos_val, partition_keys='num_predicates')


	def active_test(self, predict_fn, X_test, kernel_type="nngp"):
		pred_mean, pred_cov = predict_fn(x_test=X_test, get=kernel_type,
										 compute_cov=True)
		pred_std = np.sqrt(np.diag(pred_cov))
		pred_std = pred_std / np.max(pred_mean, 0)
		num_test = X_test.shape[0]
		pred_std = np.reshape(pred_std, newshape=(num_test,))
		std_prob = pred_std / np.sum(pred_std)
		num_select = self.budget if num_test > self.budget else num_test
		indices = random.choice(key= random.PRNGKey(10), a=num_test, shape=(num_select, ), replace=False,
								   p=std_prob) \
			if self.biased_sample else np.argsort(pred_std)[- num_select:]
		return indices

	def merge_data(self, select_indices, X_train, Y_train, X_test, Y_test):
		X_delta, Y_delta = X_test[select_indices], Y_test[select_indices]
		X_train_new = np.vstack((X_train, X_delta))
		Y_train_new = np.vstack((Y_train, Y_delta))
		num_test = X_test.shape[0]
		indices = onp.array(list(range(num_test)))
		keep_indices = onp.setdiff1d(indices, onp.asarray(select_indices))
		X_test_new, Y_test_new = X_test[keep_indices], Y_test[keep_indices]
		return X_train_new, Y_train_new, X_test_new, Y_test_new

	def active_train(self, kernel_fn, X_train, Y_train, X_test, Y_test, X_val, Y_val, query_infos_val = None):
		print("# Initial Training samples: {}".format(X_train.shape[0]))
		predict_fn = self.train(kernel_fn, X_train, Y_train)
		self.test(predict_fn, X_val, Y_val, query_infos_val, self.kernel_type)
		for i in range(self.active_iters):
			select_indices = self.active_test(predict_fn, X_test, self.kernel_type)
			print("Active Iteration {}: Selection {}".format(i, select_indices.shape[0]))
			X_train, Y_train, X_test, Y_test = self.merge_data(select_indices, X_train, Y_train, X_test, Y_test)
			print("# Training samples: {}".format(X_train.shape[0]))
			predict_fn = self.train(kernel_fn, X_train, Y_train)
			self.test(predict_fn, X_val, Y_val, query_infos_val, self.kernel_type)
