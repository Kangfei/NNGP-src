from jax import grad
from functools import partial
import jax.numpy as np
import numpy as onp
from jax.config import config
import jax.scipy as scipy
from neural_tangents import stax
import neural_tangents as nt
from .util import load_training_schema_data
import datetime
from multiprocessing import Pool
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")


class Estimator(object):
	def __init__(self, schema_name: str,  data_path: str, train_query_path: str,
				 chunk_size : int = 64, use_aux: bool= False, q_error_threshold: float= 100.0, coef_var_threshold:float=1.0):
		self.schema_name = schema_name
		self.data_path = data_path
		self.train_query_path = train_query_path
		self.chunk_size = chunk_size
		print("loading schema and training data ... This may take seconds ...")
		X_train, Y_train, self.nngp_encoder = load_training_schema_data(schema_name, data_path, train_query_path, chunk_size, use_aux, q_error_threshold, coef_var_threshold)
		self.X_train, self.Y_train = np.asarray(X_train), np.asarray(Y_train)
		print("Building model kernel ...")
		init_fn, apply_fn, kernel_fn = stax.serial(
			stax.Dense(512), stax.Relu(),
			stax.Dense(1)
		)
		kernel_fn = nt.batch(kernel_fn,
							 device_count=0,
							 batch_size=0)
		self.predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, self.X_train,
															  self.Y_train, diag_reg=1e-3)

	def load_model(self):
		pred_mean, pred_cov = self._nngp_prediction(self.X_train)
		print(pred_mean.shape, pred_cov.shape)
		print("Model construction complete.")

	def predict(self, query_lines):
		# TODO :: parallel encoding
		start = datetime.datetime.now()

		X_test = []
		for line in query_lines:
			x = self.nngp_encoder.parse_line_without_card_then_encode(line)
			X_test.append(x)
		X_test = np.asarray(X_test)
		pred_mean, pred_cov = self._nngp_prediction(X_test)
		end = datetime.datetime.now()
		duration = (end - start).total_seconds()
		print("prediction time={} seconds".format(duration))
		pred_std = np.sqrt(np.diag(pred_cov))
		pred_mean, pred_std = pred_mean.ravel(), pred_std.ravel()
		#pred_mean = onp.asarray(pred_mean)
		#pred_std = onp.asarray(pred_std)
		#pred_mean = onp.ravel(onp.array(pred_mean)).tolist()
		#pred_std = onp.ravel(onp.array(pred_std)).tolist()
		return pred_mean, pred_std


	def _nngp_prediction(self, X_test, kernel_type="nngp", compute_cov = True):

		pred_mean, pred_cov = self.predict_fn(x_test=X_test, get=kernel_type,
									 compute_cov= compute_cov)
		return pred_mean, pred_cov

