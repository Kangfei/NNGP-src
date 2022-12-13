import os
import numpy as onp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import random
from scipy import stats
from decimal import *
import pynvml
import psutil

def make_dir(dir_str: str):
	if not os.path.exists(dir_str):
		os.makedirs(dir_str)

def show_memory_usage(cuda):
	if cuda:
		pynvml.nvmlInit()
		handle = pynvml.nvmlDeviceGetHandleByIndex(0)
		meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
		print("GPU Memory Usage:",meminfo.used)
	else:
		print('CPU Memory usage: {} GB'.format(str(float(psutil.virtual_memory().used/(1024**3)))[:5]))


def draw_kernel_heatmap(kernel_mat, output_name):
	output_dir = "./{}.eps".format(output_name)
	plt.figure(figsize=(8, 8), dpi=80)
	matplotlib.rcParams['pdf.fonttype'] = 42
	matplotlib.rcParams['ps.fonttype'] = 42
	matplotlib.rc('font', family='serif')


	ax = sns.heatmap(data=kernel_mat, xticklabels=False, yticklabels=False, cbar=False)
	#ax.set_rasterized(True)
	#ax = sns.heatmap(data=kernel_mat, xticklabels=False, yticklabels=False)
	#labels = ['2', '4', '6', '8', '10']
	nums = [50, 150, 250, 350, 450]
	#labels = ['0', '1', '2']
	#nums = [125, 250, 375]
	#labels = ['0', '1', '2', '3']
	labels = ['0', '1', '2', '3', '4']
	#nums = [100, 200, 300, 400]
	plt.xticks(nums, labels, fontsize=30)
	#plt.yticks(nums, labels, fontsize=30)
	plt.yticks([50, 450], ['low', 'high'], fontsize=30)
	plt.yticks(rotation=90)


	#plt.xlabel('# of predicates (Train)', fontsize=30)
	#plt.ylabel('Standard Deviation (Test)', fontsize=30)
	plt.ylabel('Coefficient of Variation (Test)', fontsize=30)
	#plt.ylabel('# of predicates', fontsize= 30)
	plt.xlabel('# of joins (Train)', fontsize=30)
	#plt.ylabel('# of joins', fontsize= 30)
	plt.savefig(output_dir, format='eps', bbox_inches='tight')


def draw_kernel_histogram(kernel_mat, output_name):
	kernel_mat = onp.ravel(kernel_mat)
	output_dir = "./{}.pdf".format(output_name)
	#plt.figure(figsize=(10, 10), dpi=100)
	ax = sns.histplot(data=kernel_mat, bins=100)
	plt.savefig(output_dir)

def draw_embeddings(embedding, output_name, label=None):

	output_dir = "./{}.pdf".format(output_name)
	ax = sns.scatterplot(x = embedding[:,0], y = embedding[:, 1], hue=label)
	plt.savefig(output_dir)

def draw_uncertainty(output_name, errors, uncertainty, y = None):
	errors = onp.power(2.0, errors)  # transform back from log scale
	errors = onp.ravel(errors)
	uncertainty = onp.ravel(uncertainty)
	if y is not None:
		y = onp.ravel(y)
	print("draw uncertainty figure...")
	output_dir = "./{}.pdf".format(output_name)
	matplotlib.rcParams['pdf.fonttype'] = 42
	matplotlib.rcParams['ps.fonttype'] = 42
	matplotlib.rcParams['xtick.labelsize'] = 15
	matplotlib.rcParams['ytick.labelsize'] = 15
	matplotlib.rc('font', family='serif')

	ax = sns.scatterplot(x=errors, y=uncertainty, hue=y, s=20, legend=False)
	ax.set(xscale="log")
	plt.xlabel('q-error', fontsize=20)
	#plt.ylabel('Standard Deviation', fontsize=20)
	plt.ylabel('Coefficient of Variation', fontsize=20)
	plt.savefig(output_dir, format='pdf', bbox_inches='tight')

def get_prediction_statistics(errors):
	lower, upper = onp.quantile(errors, 0.25), onp.quantile(errors, 0.75)
	print("<" * 80)
	print("Predict Result Profile of {} Queries:".format(len(errors)))
	print("Min/Max: {:.4f} / {:.4f}".format(onp.min(errors), onp.max(errors)))
	print("Mean: {:.4f}".format(onp.mean(errors)))
	print("Median: {:.4f}".format(onp.median(errors)))
	print("25%/75% Quantiles: {:.4f} / {:.4f}".format(lower, upper))
	print("5%/95% Quantiles: {:.4f} / {:.4f}".format(onp.quantile(errors, 0.05), onp.quantile(errors, 0.95)))
	print(">" * 80)
	error_median = abs(upper - lower)
	return error_median

class PredictionStatistics(object):
	def __init__(self):
		self.keys = ['num_table', 'num_joins', 'num_predicates']

	def get_prediction_details(self, errors, query_infos = None, partition_keys=''):
		if query_infos is None or not partition_keys:
			self.get_prediction_statistics(errors)
			return
		partition_keys = partition_keys.strip().split(',')
		partition_keys = [key.strip() for key in partition_keys]
		for key in partition_keys:
			assert key in self.keys, "Unsupported partition key!"

		partition_errors = {}
		for error, query_info in zip(errors.tolist(), query_infos):
			query_attrs = tuple( getattr(query_info, key) for key in partition_keys)
			if query_attrs not in partition_errors.keys():
				partition_errors[query_attrs] = []
			partition_errors[query_attrs].append(error)

		# shrink the result display size
		tmp_partition_errors = {}
		if len(partition_errors) > 6:
			tmp_partition_errors_list = [(query_attrs, partition_errors[query_attrs])
									for query_attrs in sorted(partition_errors.keys())]
			for i, (query_attrs, errors) in enumerate(tmp_partition_errors_list):
				if i % 2 == 0 and i < len(tmp_partition_errors_list) - 1:
					continue
				elif i % 2 == 1:
					errors += tmp_partition_errors_list[i - 1][1]
					tmp_partition_errors[query_attrs] = errors
				else:
					tmp_partition_errors[query_attrs] = errors
			partition_errors = tmp_partition_errors

		for query_attrs in sorted(partition_errors.keys()):
			info_str = ["{}={}".format(key, attr) for key, attr in zip(partition_keys, list(query_attrs))]
			info_str = 'Query attributes:' + ','.join(info_str)
			print(info_str)
			print('# Queries = {}'.format(len(partition_errors[query_attrs])))
			error = onp.array(partition_errors[query_attrs])
			self.get_prediction_statistics(error)



	def get_prediction_statistics(self, errors):
		errors = onp.power(2.0, errors) # transform back from log scale
		lower, upper = onp.quantile(errors, 0.25), onp.quantile(errors, 0.75)
		print("<" * 80)
		print("Predict Result Profile of {} Queries:".format(len(errors)))
		print("Min/Max: {:.15f} / {:.15f}".format(onp.min(errors), onp.max(errors)))
		print("Mean: {:.8f}".format(onp.mean(errors)))
		print("Median: {:.8f}".format(onp.median(errors)))
		print("25%/75% Quantiles: {:.8f} / {:.8f}".format(lower, upper))
		print("5%/95% Quantiles: {:.8f} / {:.8f}".format(onp.quantile(errors, 0.05), onp.quantile(errors, 0.95)))
		#plot_str = "lower whisker={:.10f}, \nlower quartile={:.10f}, \nmedian={:.10f}, \nupper quartile={:.10f}, \nupper whisker={:.10f},"\
		#	.format(onp.min(errors), lower, onp.median(errors), upper, onp.max(errors))
		#print(plot_str)
		print(">" * 80)
		error_median = abs(upper - lower)
		return error_median

	def get_permutation_index(self, query_infos, perm_keys=''):
		# return a numpy array as the permutation index based on the perm_keys
		num_instances = len(query_infos)
		if not perm_keys:
			return onp.array(list(range(num_instances)))

		partition_query_indices = self.get_partitioned_indices(query_infos, part_keys= perm_keys)

		permutation = []
		for query_attrs in sorted(partition_query_indices.keys()):
			for idx in partition_query_indices[query_attrs]:
				permutation.append(idx)
		return onp.array(permutation)

	def get_permutation_data(self, X, query_infos, perm_keys):
		num_instances = len(X) if isinstance(X, list) else X.shape[0]
		assert num_instances == len(query_infos), "Data size inconsistent with query info!"
		permutation = self.get_permutation_index(query_infos, perm_keys)
		if isinstance(X, list):
			X = [X[idx] for idx in permutation.tolist()]
		else:
			X = X[permutation]
		return X

	def get_partitioned_data(self, X, query_infos, part_keys):
		num_instances = len(X) if isinstance(X, list) else X.shape[0]
		assert num_instances == len(query_infos), "Data size inconsistent with query info!"
		partition_query_indices = self.get_partitioned_indices(query_infos, part_keys)

		partitioned_X = []
		for query_attrs in sorted(partition_query_indices.keys()):
			x = [X[idx] for idx in partition_query_indices[query_attrs]]
			if not isinstance(X, list):
				x = onp.asarray(x)
			partitioned_X.append(x)
		return partitioned_X

	def get_partitioned_indices(self, query_infos, part_keys):
		part_keys = part_keys.strip().split(',')
		part_keys = [key.strip() for key in part_keys]
		for key in part_keys:
			assert key in self.keys, "Unsupported partition key!"
		partition_query_indices = {}
		for i, query_info in enumerate(query_infos):
			query_attrs = tuple(getattr(query_info, key) for key in part_keys)
			if query_attrs not in partition_query_indices.keys():
				partition_query_indices[query_attrs] = []
			partition_query_indices[query_attrs].append(i)
		return partition_query_indices


def uneven_train_test_split(X, Y, all_query_infos, skew_split_keys, train_frac = 0.6, skew_ratio = 0.5, seed=10):
	"""
	split train/test data by train_frac and unevenly split the train data by attributes in skew_split_keys
	"""

	random.seed(seed)
	pred_stat = PredictionStatistics()
	partition_query_indices = pred_stat.get_partitioned_indices(all_query_infos, part_keys=skew_split_keys)
	num_partitioned_set = len(partition_query_indices)
	tmp_partitioned_query_indices_train = {}
	X_test, Y_test, query_infos_test = [], [], []
	X_train, Y_train, query_infos_train = [], [], []

	for key in sorted(partition_query_indices.keys()):
		# shuffle partitioned indices
		random.shuffle(partition_query_indices[key])

		num_train = int(len(partition_query_indices[key]) * train_frac)

		partition_X_test = [ X[idx] for idx in partition_query_indices[key][num_train:] ]
		partition_Y_test = [ Y[idx] for idx in partition_query_indices[key][num_train:] ]
		partition_query_infos_test = [ all_query_infos[idx] for idx in partition_query_indices[key][num_train:] ]
		X_test += partition_X_test
		Y_test += partition_Y_test
		query_infos_test += partition_query_infos_test
		tmp_partitioned_query_indices_train[key] = partition_query_indices[key][:num_train]

	for i, key in enumerate(sorted(tmp_partitioned_query_indices_train.keys())):
		if num_partitioned_set % 2 == 0:
			select_ratio = skew_ratio if i < int(num_partitioned_set / 2) else Decimal(1) - Decimal(skew_ratio)
		else:
			if i < int(num_partitioned_set / 2):
				select_ratio = skew_ratio
			elif i == int(num_partitioned_set / 2):
				select_ratio = 0.5
			else:
				select_ratio = float(Decimal(1) - Decimal(skew_ratio))
		num_train = int(len(tmp_partitioned_query_indices_train[key]) * select_ratio)
		print(select_ratio, num_train)
		partition_X_train = [ X[idx] for idx in tmp_partitioned_query_indices_train[key][: num_train] ]
		partition_Y_train = [ Y[idx] for idx in tmp_partitioned_query_indices_train[key][: num_train] ]
		partition_query_infos_train = [all_query_infos[idx] for idx in tmp_partitioned_query_indices_train[key][: num_train]]
		X_train += partition_X_train
		Y_train += partition_Y_train
		query_infos_train += partition_query_infos_train
	Y_train, Y_test = onp.asarray(Y_train), onp.asarray(Y_test)
	if isinstance(X, onp.ndarray):
		X_train, X_test = onp.array(X_train), onp.array(X_test)
	return X_train, Y_train, query_infos_train, X_test, Y_test, query_infos_test, None, None, None


def train_test_val_split(X, Y, train_frac=0.6, test_frac=0.2, seed=10, all_query_infos= None, max_num_train = None):
	# default split train/test/val: 6/2/2
	num_instances = X.shape[0]
	print("# instances = {}".format(num_instances))
	num_train, num_test = int(train_frac * num_instances), int(test_frac * num_instances)
	indices = list(range(num_instances))
	random.seed(seed)
	random.shuffle(indices)
	X, Y = X[indices, :], Y[indices, :]
	if all_query_infos is not None:
		all_query_infos = [ all_query_infos[idx] for idx in indices]
	X_train, Y_train = X[:num_train, :], Y[:num_train, :]
	X_test, Y_test = X[num_train: num_train + num_test, :], Y[num_train: num_train + num_test, :]
	X_val = X[num_train + num_test :, :] if train_frac + test_frac < 1 else None
	Y_val = Y[num_train + num_test :, :] if train_frac + test_frac < 1 else None
	query_infos_train = all_query_infos[:num_train] if all_query_infos is not None else None
	query_infos_test = all_query_infos[num_train : num_train + num_test] if all_query_infos is not None else None
	query_infos_val = all_query_infos[num_train + num_test:] if all_query_infos is not None and train_frac + test_frac < 1 else None
	if max_num_train is not None and max_num_train <= num_train:
		query_infos_train = query_infos_train[:max_num_train]
		X_train = X_train[:max_num_train]
		Y_train = Y_train[:max_num_train]
	return X_train, Y_train, query_infos_train, X_test, Y_test, query_infos_test, X_val, Y_val, query_infos_val


def calibration_plot(Y_test, means, stds, num_intervals= 10):
	Y_test = onp.ravel(Y_test).tolist()
	means = onp.ravel(means).tolist()
	stds = onp.ravel(stds).tolist()
	num_queries = len(Y_test)
	level_step = 1.0 / num_intervals
	levels = [level_step * i for i in range(num_intervals + 1)]
	cal_cnt_dict = {level : 0.0 for level in levels}
	for output, mean, std in zip(Y_test, means, stds):
		for level in levels:
			(lower, upper) = stats.norm.interval(alpha=level, loc=mean, scale= std)
			if lower <= output <= upper:
				cal_cnt_dict[level] += 1.0
	print("<" * 80)
	print("Calibration Result:")
	for level, cnt in cal_cnt_dict.items():
		print("Expected/Observed Confidence Level={}/{}".format(level, cnt/num_queries))
	print(">" * 80)


def transform_category_encoding(df, col_types):
	for col_idx, col_name in enumerate(df.columns):
		if col_types[col_idx] == 'categorical':
			#print(df[col_name])
			df[col_name] = pd.Categorical(df[col_name]).codes
	return df

