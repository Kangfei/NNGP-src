import numpy as np
import random
from torch.utils.data.dataset import Dataset
import math
import torch


def train_test_val_split(X, Y, train_frac=0.6, test_frac=0.2, seed=10, all_query_infos= None):
	# default split train/test/val: 6/2/2
	num_instances = len(X)
	print("# instances = {}".format(num_instances))
	num_train, num_test = int(train_frac * num_instances), int(test_frac * num_instances)
	indices = list(range(num_instances))
	random.seed(seed)
	random.shuffle(indices)
	X = [ X[idx] for idx in indices]
	Y = Y[indices, :]
	if all_query_infos is not None:
		all_query_infos = [ all_query_infos[idx] for idx in indices]
	X_train, Y_train = X[:num_train], Y[:num_train, :]
	X_test, Y_test = X[num_train: num_train + num_test], Y[num_train: num_train + num_test, :]
	X_val = X[num_train + num_test :] if train_frac + test_frac < 1 else None
	Y_val = Y[num_train + num_test :, :] if train_frac + test_frac < 1 else None
	query_infos_train = all_query_infos[:num_train] if all_query_infos is not None else None
	query_infos_test = all_query_infos[num_train: num_train + num_test] if all_query_infos is not None else None
	query_infos_val = all_query_infos[num_train + num_test: ] if all_query_infos is not None and train_frac + test_frac < 1 else None
	return X_train, Y_train, query_infos_train, X_test, Y_test, query_infos_test, X_val, Y_val, query_infos_val

class MSCNDataset(Dataset):
	def __init__(self, X, Y, join_query, max_classes=10):
		self.join_query = join_query
		self.label_base = 10
		self.max_classes = max_classes
		self.pred_X, self.join_X = self.set_padding(X)
		self.Y = Y

	def __len__(self):
		return self.Y.shape[0]

	def set_padding(self, X):
		left_pad_pred_size, right_pad_pred_size, pad_pred_size, pad_join_size = 0, 0, 0, 0
		left_pred_X, right_pred_X, pred_X, join_X = [], [], [], []
		if self.join_query:
			for left_pred_x, right_pred_x, join_x in X:
				left_pad_pred_size = max(left_pad_pred_size, left_pred_x.shape[0])
				right_pad_pred_size = max(right_pad_pred_size, right_pred_x.shape[0])
				pad_join_size = max(pad_join_size, join_x.shape[0])
			for left_pred_x, right_pred_x, join_x in X:
				left_pred_x = np.pad(left_pred_x, ((0, left_pad_pred_size - left_pred_x.shape[0]), (0, 0)), 'constant') # use zero pad
				right_pred_x = np.pad(right_pred_x, ((0, right_pad_pred_size - right_pred_x.shape[0]), (0, 0)),
									 'constant')  # use zero pad
				join_x = np.pad(join_x, ((0, pad_join_size - join_x.shape[0]), (0, 0)), 'constant')
				left_pred_X.append(left_pred_x)
				right_pred_X.append(right_pred_x)
				join_X.append(join_x)
			left_pred_X = np.array(left_pred_X)
			right_pred_X = np.array(right_pred_X)
			join_X = np.array(join_X)
			return (left_pred_X, right_pred_X), join_X
		else:
			for pred_x in X:
				pad_pred_size = max(pad_pred_size, pred_x.shape[0])
				#print(pad_pred_size)
			for pred_x in X:
				pred_x = np.pad(pred_x, ((0, pad_pred_size - pred_x.shape[0]), (0, 0)), 'constant')  # use zero pad
				pred_X.append(pred_x)
			pred_X = np.array(pred_X)
			#print(pred_X.shape)
			return pred_X, None

	def __getitem__(self, index):
		y = self.Y[index]
		idx = math.ceil(math.log(math.pow(2, y), self.label_base))
		idx = self.max_classes - 1 if idx >= self.max_classes else idx
		label = torch.tensor(idx, dtype=torch.long)
		y = torch.FloatTensor(y)

		if self.join_query:
			left_pred_x, right_pred_x, join_x = self.pred_X[0][index], self.pred_X[1][index], self.join_X[index]
			left_pred_x = torch.FloatTensor(left_pred_x)
			right_pred_x = torch.FloatTensor(right_pred_x)
			join_x = torch.FloatTensor(join_x)
			return left_pred_x, right_pred_x, join_x, y, label
		else:
			pred_x = self.pred_X[index]
			pred_x = torch.FloatTensor(pred_x)
			return pred_x, y, label


class MultiJoinMSCNDataset(Dataset):
	def __init__(self, X, Y, max_classes = 10):
		# x = (table_x, pred_x, join_x)
		self.label_base = 10
		self.max_classes = max_classes
		self.table_X, self.pred_X, self.join_X = self.set_padding(X)
		self.Y = Y

	def __len__(self):
		return self.Y.shape[0]

	def set_padding(self, X):
		pad_table_size, pad_pred_size, pad_join_size = 0, 0, 0
		table_X, pred_X, join_X = [], [], []
		for (table_x, pred_x, join_x) in X:
			pad_table_size = max(pad_table_size, table_x.shape[0])
			pad_pred_size = max(pad_pred_size, pred_x.shape[0])
			pad_join_size = max(pad_join_size, join_x.shape[0])

		for (table_x, pred_x, join_x) in X:
			table_x = np.pad(table_x, ((0, pad_table_size - table_x.shape[0]), (0, 0)), 'constant')
			pred_x = np.pad(pred_x, ((0, pad_pred_size - pred_x.shape[0]), (0, 0)), 'constant')
			join_x = np.pad(join_x, ((0, pad_join_size - join_x.shape[0]), (0, 0)), 'constant')
			table_X.append(table_x)
			pred_X.append(pred_x)
			join_X.append(join_x)
		table_X = np.array(table_X)
		pred_X = np.array(pred_X)
		join_X = np.array(join_X)
		return table_X, pred_X, join_X

	def __getitem__(self, index):
		y = self.Y[index]
		idx = math.ceil(math.log(math.pow(2, y), self.label_base))
		idx = self.max_classes - 1 if idx >= self.max_classes else idx
		label = torch.tensor(idx, dtype=torch.long)
		y = torch.FloatTensor(y)
		table_x, pred_x, join_x = self.table_X[index], self.pred_X[index], self.join_X[index]
		table_x = torch.FloatTensor(table_x)
		pred_x = torch.FloatTensor(pred_x)
		join_x = torch.FloatTensor(join_x)
		return table_x, pred_x, join_x, y, label

