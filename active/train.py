import jax.numpy as np
import random

def get_prediction_statistics(errors):
	lower, upper = np.quantile(errors, 0.25), np.quantile(errors, 0.75)
	print("<" * 80)
	print("Predict Result Profile of {} Queries:".format(len(errors)))
	print("Min/Max: {:.4f} / {:.4f}".format(np.min(errors), np.max(errors)))
	print("Mean: {:.4f}".format(np.mean(errors)))
	print("Median: {:.4f}".format(np.median(errors)))
	print("25%/75% Quantiles: {:.4f} / {:.4f}".format(lower, upper))
	print("5%/95% Quantiles: {:.4f} / {:.4f}".format(np.quantile(errors, 0.05), np.quantile(errors, 0.95)))
	print(">" * 80)
	error_median = abs(upper - lower)
	return error_median


def train_test_val_split(X, Y, train_frac=0.6, test_frac=0.2, seed=4):
	# default split train/test/val: 6/2/2
	num_instances = X.shape[0]
	num_train, num_test = int(train_frac * num_instances), int(test_frac * num_instances)
	indices = list(range(num_instances))
	random.seed(seed)
	random.shuffle(indices)
	X, Y = X[indices, :], Y[indices, :]
	X_train, Y_train = X[:num_train, :], Y[:num_train, :]
	X_test, Y_test = X[num_train: num_train + num_test, :], Y[num_train: num_train + num_test, :]
	X_val, Y_val = X[num_train + num_test:, :], Y[num_train+ num_test :, :]
	return X_train, Y_train, X_test, Y_test, X_val, Y_val

