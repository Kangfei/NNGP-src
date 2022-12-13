from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import os
import datetime
import random
import sys
import jax.numpy as np
from jax.api import jit
from jax.lib import xla_bridge
from jax.config import config
from neural_tangents import stax
import neural_tangents as nt
from active.ActiveLearner import ActiveLearner
from util import train_test_val_split
import datasets
import schemas


config.update("jax_enable_x64", True)


def main(args):
	#X, Y, all_query_infos = datasets.load_training_data(args)
	X, Y, all_query_infos = schemas.load_training_schema_data(args)
	num_queries = X.shape[0]
	print("number of query: {}".format(num_queries))
	X_train, Y_train, query_infos_train, X_test, Y_test, query_infos_test, X_val, Y_val, query_infos_val = \
		train_test_val_split(X, Y, train_frac=0.2, test_frac=0.6, all_query_infos=all_query_infos)
	# X_train, Y_train, query_infos_train, X_test, Y_test, query_infos_test, X_val, Y_val, query_infos_val = \
	#	uneven_train_test_split(X, Y, all_query_infos=all_query_infos, skew_split_keys='num_table', train_frac = 0.8, skew_ratio = 0.8)

	X_train, Y_train = np.asarray(X_train), np.asarray(Y_train)
	X_test, Y_test = np.asarray(X_test), np.asarray(Y_test)
	X_val = np.asarray(X_val) if X_val is not None else None
	Y_val = np.asarray(Y_val) if Y_val is not None else None

	print(X_train.shape, X_test.shape)
	print(Y_train.shape, Y_test.shape)

	# Define the model
	init_fn, apply_fn, kernel_fn = stax.serial(
		stax.Dense(512), stax.Relu(),
		stax.Dense(1)
	)
	"""
	init_fn, apply_fn, kernel_fn = stax.serial(
		stax.Dense(512, W_std=1.5, b_std=0.05), stax.Relu(),
		stax.Dense(1, W_std=1.5, b_std=0.05)
	)
	"""
	active_learner = ActiveLearner(args)
	active_learner.active_train(kernel_fn, X_train, Y_train, X_test, Y_test, X_val, Y_val, query_infos_val)


if __name__ == "__main__":
	parser = ArgumentParser("NNGP estimator", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
	parser.add_argument('--kernel_type', type=str, default="nngp",
						help='Selected Queries budget Per Iteration.')
	parser.add_argument("--chunk_size", default=10, type=int, help="dimension of factorized encoding")
	parser.add_argument("--feat_encode", type=str, default='dnn-encoder', help='dnn-encoder,one-hot')
	parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')

	parser.add_argument("--biased_sample", default=True, type=bool,
						help="Enable Biased sampling for test set selection")
	parser.add_argument('--active_iters', type=int, default=3,
						help='Num of iterators of active learning.')
	parser.add_argument('--budget', type=int, default=1000,
						help='Selected Queries budget Per Iteration.')

	# input dir
	# yelp-user
	#parser.add_argument("--relations", type=str, default='yelp-user') # 'yelp-user'
	#parser.add_argument("--names", type=str, default='user')
	#parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/yelp-user_100')

	#parser.add_argument("--relations", type=str, default='yelp-review,yelp-user')  # 'reviewer,user'
	#parser.add_argument("--names", type=str, default='review,user')
	#parser.add_argument("--query_path", type=str,
	#					default='/home/kfzhao/PycharmProjects/NNGP/queryset/join_review_user_100_2')

	#parser.add_argument("--relations", type=str, default='forest,forest')  # 'forest,forest'
	#parser.add_argument("--names", type=str, default='forest1,forest2')
	#parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/join_forest1_forest2')

	parser.add_argument("--relations", type=str, default='forest')
	parser.add_argument("--names", type=str, default='forest')
	parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/forest_data')

	#parser.add_argument("--relations", type=str, default='higgs')
	#parser.add_argument("--names", type=str, default='higgs')
	#parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/higgs')

	parser.add_argument("--schema_name", type=str, default='tpch', help='yelp, tpcds, tpch')

	# parser.add_argument("--query_path", type=str, default="/home/kfzhao/PycharmProjects/NNGP/queryset/join_business_review_user_10_data_centric")
	# parser.add_argument("--query_path", type=str,  default="/home/kfzhao/PycharmProjects/NNGP/queryset/join_store_sales_store_item_customer_promotion_10_data_centric_427")
	parser.add_argument("--query_path", type=str,  default="/home/kfzhao/PycharmProjects/NNGP/queryset/join_lineitem_part_orders_supplier_10_data_centric_427")
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