from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import os
import datetime
import sys
from absl import app
from absl import flags
from jax import grad
from functools import partial
import jax.numpy as np
import numpy as onp
# from jax.api import jit
import jax.random as rand
from jax import vmap
from jax.lib import xla_bridge
from jax.config import config
import jax.scipy as scipy
from neural_tangents import stax
import neural_tangents as nt
import datasets
import schemas
from util import draw_uncertainty, calibration_plot, PredictionStatistics, draw_kernel_heatmap, show_memory_usage
from util import uneven_train_test_split, train_test_val_split

config.update("jax_enable_x64", True)
#os.environ['TF_XLA_FLAGS']="--xla_gpu_cuda_data_dir=/usr/local/cuda"

pred_stat = PredictionStatistics()

def permute_kernel_matrix(kernel_mat, query_infos, perm_keys):
	assert kernel_mat.shape[0] == len(query_infos), "Permute length inconsistent with query info!"
	perm = pred_stat.get_permutation_index(query_infos, perm_keys)
	N = kernel_mat.shape[0]
	# permute the kernel matrix by array perm
	for i in range(N):
		kernel_mat[:, i] = kernel_mat[perm, i]
	for i in range(N):
		kernel_mat[i, :] = kernel_mat[i, perm]
	return kernel_mat

def permute_train_test_kernel_matrix(kernel_mat, train_query_infos, perm_keys, pred_std):
	assert kernel_mat.shape[0] == len(train_query_infos), "Permute length inconsistent with train query info!"
	assert kernel_mat.shape[1] == pred_std.shape[0], "Permute length inconsistent with test std!"
	train_perm = pred_stat.get_permutation_index(train_query_infos, perm_keys)
	test_perm = onp.argsort(pred_std)
	kernel_mat = kernel_mat[test_perm]
	kernel_mat = kernel_mat[:, train_perm]
	return kernel_mat


def draw_kernel_histogram(kernel_mat, output_name):
	kernel_mat = np.ravel(kernel_mat)
	output_dir = "./{}.pdf".format(output_name)
	import seaborn as sns
	import matplotlib.pyplot as plt
	#plt.figure(figsize=(10, 10), dpi=100)
	ax = sns.histplot(data=kernel_mat, bins=100)
	plt.grid(b=None)
	plt.savefig(output_dir)

def GP_train_and_test(X_train, Y_train, X_test, Y_test, query_infos_train = None, query_infos_test= None):
	numpts = Y_train.shape[0]
	key = rand.PRNGKey(0)
	eye = np.eye(numpts)

	def cov_map(cov_func, xs, xs2=None):
		"""Compute a covariance matrix from a covariance function and data points.
		Args:
			cov_func: callable function, maps pairs of data points to scalars.
			xs: array of data points, stacked along the leading dimension.
		Returns:
			A 2d array `a` such that `a[i, j] = cov_func(xs[i], xs[j])`.
		"""
		if xs2 is None:
			return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs)
		else:
			return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs2).T

	def softplus(x):
		return np.logaddexp(x, 0.)
	# Note, writing out the vectorized form of the identity
	# ||x-y||^2 = <x-y,x-y> = ||x||^2 + ||y||^2 - 2<x,y>
	# for computing squared distances would be more efficient (but less succinct).
	def exp_quadratic(x1, x2):
		return np.exp(-np.sum((x1 - x2)**2))

	def gp(params, x, y, xtest=None, compute_marginal_likelihood=False):
		noise = softplus(params['noise'])
		amp = softplus(params['amplitude'])
		ls = softplus(params['lengthscale'])
		ymean = np.mean(y)
		y = y - ymean
		x = x / ls
		train_cov = amp*cov_map(exp_quadratic, x) + eye * (noise + 1e-6)
		chol = scipy.linalg.cholesky(train_cov, lower=True)
		kinvy = scipy.linalg.solve_triangular(chol.T, scipy.linalg.solve_triangular(chol, y, lower=True))
		if compute_marginal_likelihood:
			log2pi = np.log(2. * 3.1415)
			ml = np.sum(
				-0.5 * np.dot(y.T, kinvy) -
				np.sum(np.log(np.diag(chol))) -
				(numpts / 2.) * log2pi)
			ml -= np.sum(-0.5 * np.log(2 * 3.1415) - np.log(amp)**2) # lognormal prior
			return -ml
		if xtest is not None:
			xtest = xtest / ls
		cross_cov = amp*cov_map(exp_quadratic, x, xtest)
		mu = np.dot(cross_cov.T, kinvy) + ymean
		v = scipy.linalg.solve_triangular(chol, cross_cov, lower=True)
		var = (amp * cov_map(exp_quadratic, xtest) - np.dot(v.T, v))
		return mu, var

	marginal_likelihood = partial(gp, compute_marginal_likelihood=True)
	predict = partial(gp, compute_marginal_likelihood=False)
	grad_fun = jit(grad(marginal_likelihood))

	# Covariance hyperparameters to be learned
	params = {"amplitude": np.zeros((1, 1)),
            "noise": np.zeros((1, 1)) - 5.,
            "lengthscale": np.zeros((1, 1))}
	momentums = dict([(k, p * 0.) for k, p in params.items()])
	scales = dict([(k, p * 0. + 1.) for k, p in params.items()])

	lr = 0.01  # Learning rate
	def train_step(params, momentums, scales, x, y):
		grads = grad_fun(params, x, y)
		for k in (params):
			momentums[k] = 0.9 * momentums[k] + 0.1 * grads[k][0]
			scales[k] = 0.9 * scales[k] + 0.1 * grads[k][0]**2
			params[k] -= lr * momentums[k]/np.sqrt(scales[k] + 1e-5)
		return params, momentums, scales

	for i in range(10):
		params, momentums, scales = train_step(params, momentums, scales, X_train, Y_train)
		if i % 1 == 0:
			ml = marginal_likelihood(params, X_train, Y_train)
			print("Step: %d, neg marginal likelihood: %f" % (i, ml))

	start = datetime.datetime.now()
	pred_mean, var = predict(params, X_train, Y_train, X_test)
	end = datetime.datetime.now()
	duration = (end - start).total_seconds()
	print('Kernel construction in %s seconds.' % duration)
	start = datetime.datetime.now()
	pred_mean, var = predict(params, X_train, Y_train, X_test)
	end = datetime.datetime.now()
	duration = (end - start).total_seconds()
	print('GP Inference in %s seconds.' % duration)
	std = np.sqrt(np.diag(var))
	errors = onp.ravel(onp.array(pred_mean - Y_test))
	pred_stat.get_prediction_details(errors, query_infos_test, partition_keys='num_predicates')


def NNGP_train_and_test(args, X_train, Y_train, X_test, Y_test, query_infos_train = None, query_infos_test= None):

	def prediction(pred_fn, X_test, kernel_type="nngp", compute_cov = True):

		pred_mean, pred_cov = pred_fn(x_test=X_test, get=kernel_type,
									 compute_cov= compute_cov)
		return pred_mean, pred_cov

	init_fn, apply_fn, kernel_fn = stax.serial(
		stax.Dense(512), stax.Relu(),
		stax.Dense(1)
	)

	kernel_fn = nt.batch(kernel_fn,
						 device_count=0,
						 batch_size=0)
	show_memory_usage(cuda=args.cuda)
	start = datetime.datetime.now()
	predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X_train,
														  Y_train, diag_reg=1e-3)
	end = datetime.datetime.now()
	show_memory_usage(cuda=args.cuda)
	duration = (end - start).total_seconds()
	print('Kernel construction in %s seconds.' % duration)

	pred_mean, pred_cov = prediction(predict_fn, X_test, kernel_type=args.kernel_type)
	show_memory_usage(cuda=args.cuda)
	pred_std = np.sqrt(np.diag(pred_cov))
	show_memory_usage(cuda=args.cuda)


	mse = np.sum(np.power(pred_mean -Y_test, 2))
	#print(mse)
	print("Mean Square Error: {}".format(mse))


	#Obtain the inference time
	print(X_test.shape, Y_test.shape)
	start = datetime.datetime.now()
	pred_mean, pred_cov = prediction(predict_fn, X_test, kernel_type=args.kernel_type)
	end = datetime.datetime.now()
	duration = (end - start).total_seconds()
	print("Inference time={} seconds".format(duration))


	errors = onp.ravel(onp.array(pred_mean - Y_test))
	pred_std = onp.ravel(onp.array(pred_std))
	outputs = onp.ravel(onp.array(pred_mean))
	Y_test = onp.ravel(onp.array(Y_test))

	pred_stat.get_prediction_details(errors, query_infos_test, partition_keys='num_table')
	"""
	all_Y_test = pred_stat.get_partitioned_data(Y_test, query_infos_test, part_keys='num_table')
	all_outputs = pred_stat.get_partitioned_data(outputs, query_infos_test, part_keys='num_table')
	all_pred_std = pred_stat.get_partitioned_data(pred_std, query_infos_test, part_keys='num_table')
	for (Y_test, outputs, pred_std) in zip(all_Y_test, all_outputs, all_pred_std):
		calibration_plot(Y_test, outputs, pred_std)
	
	pred_std = pred_std / np.max(outputs, 0)
	#draw_uncertainty("tpcds_uncertainty", errors, pred_std, Y_test)


	print("compute train kernel function:")
	kernel = kernel_fn(X_test[:500], X_train[:500], args.kernel_type)
	kernel = onp.array(kernel)
	print(kernel.shape)
	kernel = permute_train_test_kernel_matrix(kernel_mat=kernel, train_query_infos=query_infos_train[:500], perm_keys='num_table', pred_std=pred_std[:500])
	#kernel = permute_kernel_matrix(kernel, query_infos_train[:500], perm_keys='num_table')
	draw_kernel_heatmap(kernel, "tpcds_train_test_kernel")
	"""

def main(args):
	if(not args.join_query):
		X, Y, all_query_infos = datasets.load_training_data(args)
	else:
		X, Y, all_query_infos = schemas.load_training_schema_data(args)
	num_queries = X.shape[0]
	print("number of query: {}".format(num_queries))
	X_train, Y_train, query_infos_train, X_test, Y_test, query_infos_test, X_val, Y_val, query_infos_val =\
		train_test_val_split(X, Y, train_frac=0.6, test_frac=0.2, all_query_infos= all_query_infos)
	#X_train, Y_train, query_infos_train, X_test, Y_test, query_infos_test, X_val, Y_val, query_infos_val = \
	#	uneven_train_test_split(X, Y, all_query_infos=all_query_infos, skew_split_keys='num_predicates', train_frac = 0.8, skew_ratio = 0.2)

	X_train, Y_train = np.asarray(X_train), np.asarray(Y_train)
	X_test, Y_test = np.asarray(X_test), np.asarray(Y_test)
	X_val = np.asarray(X_val) if X_val is not None else None
	Y_val = np.asarray(Y_val) if Y_val is not None else None

	print(X_train.shape, X_test.shape)
	print(Y_train.shape, Y_test.shape)
	if args.kernel_type == 'gp':
		GP_train_and_test(X_train, Y_train, X_test, Y_test, query_infos_train, query_infos_test)
	else:
		NNGP_train_and_test(args, X_train, Y_train, X_test, Y_test, query_infos_train, query_infos_test)




if __name__ == "__main__":
	parser = ArgumentParser("NNGP/NTK estimator", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
	parser.add_argument("--chunk_size", default=64, type=int, help="dimension of factorized encoding")
	parser.add_argument("--kernel_type", type=str, default='nngp', help='nngp, ntk')
	parser.add_argument("--feat_encode", type=str, default='dnn-encoder', help='dnn-encoder,one-hot')
	parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
	# input dir
	parser.add_argument("--relations", type=str, default='forest')
	parser.add_argument("--names", type=str, default='forest')
	#parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/forest')
	#parser.add_argument("--data_path", type=str, default='/home/kfzhao/data/UCI/')

	#parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/join_store_sales_store_item_customer_promotion_10_data_centric_805_2G')
	#parser.add_argument("--data_path", type=str, default='/home/kfzhao/data/rdb/TPCDS_2Gclean')

	parser.add_argument("--query_path", type=str,
						default='/home/kfzhao/PycharmProjects/NNGP/queryset/join_title_cast_info_movie_info_movie_companies_movie_info_idx_movie_keyword_10_data_centric_815_FP')
	parser.add_argument("--data_path", type=str, default='/home/kfzhao/data/rdb/imdb_clean')

	#parser.add_argument("--relations", type=str, default='higgs')
	#parser.add_argument("--names", type=str, default='higgs')
	#parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/higgs')

	#parser.add_argument("--relations", type=str, default='yelp-review,yelp-user')  # 'yelp-user'
	#parser.add_argument("--names", type=str, default='review,user')
	#parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/join_review_user_100_2')

	#parser.add_argument("--relations", type=str, default='forest,forest')  # 'forest,forest'
	#parser.add_argument("--names", type=str, default='forest1,forest2')
	#parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/join_forest1_forest2')

	parser.add_argument("--schema_name", type=str, default='imdb_simple', help='yelp, tpcds, tpch')

	#parser.add_argument("--query_path", type=str, default="/home/kfzhao/PycharmProjects/NNGP/queryset/join_business_review_user_10_data_centric") # yelp
	#parser.add_argument("--query_path", type=str,  default="/home/kfzhao/PycharmProjects/NNGP/queryset/join_store_sales_store_item_customer_promotion_10_data_centric_422") #tpcds
	#parser.add_argument("--query_path", type=str,  default="/home/kfzhao/PycharmProjects/NNGP/queryset/join_lineitem_part_orders_supplier_10_data_centric_427") # tpch
	args = parser.parse_args()
	args.cuda = not args.no_cuda and xla_bridge.get_backend().platform == 'gpu'

	if args.cuda:
		config.update("jax_platform_name", "gpu")
	else:
		config.update("jax_platform_name", "cpu")

	relations = args.relations.split(',')
	args.join_query = True if len(relations) > 1 else False
	print(args)
	main(args)
