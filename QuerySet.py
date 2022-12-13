import os
import numpy as np

class QuerySet(object):
	def __init__(self, query_dir: str, dataset: str, df):
		self.df =df
		self.query_dir = query_dir
		self.dataset = dataset
		self.query_path = os.path.join(query_dir, dataset)
		self.num_cols = len(df.columns)
		self.all_cols = df.columns
		self.all_col_ranges = np.zeros(shape=(self.num_cols, 2))
		for i in range(self.num_cols):
			single_col_df = self.df.iloc[:, i]
			single_col_df = single_col_df.sort_values()
			self.all_col_df.append(single_col_df)
			self.all_col_ranges[i][0] = single_col_df.min()
			self.all_col_ranges[i][1] = single_col_df.max()


	def parse_line(self, line: str):
		pred_str, card = line.split("@")[0].strip(), int(line.split("@")[1].strip())
		predicates = pred_str.split("#")
		pred_list = []
		for predicate in predicates:
			col_name, upper, lower = predicate.split(",")[0].strip(), float(predicate.split(",")[1].strip()), float(predicate.split(",")[2].strip())
			col_idx = ord(col_name) - 65
			pred_list.append((col_idx, upper, lower))
		return pred_list, card

	def load_queries(self):
		sub_dirs = os.listdir(self.query_path)
		all_queries = []
		all_cards = []
		for sub_dir in sub_dirs:
			with open(os.path.join(self.query_path, sub_dir), "r") as in_file:
				for line in in_file:
					pred_list, card = self.parse_line(line)
					all_queries.append(pred_list)
					all_cards.append(card)
				in_file.close()
		return all_queries, all_cards

	def transform_to_array(self, all_queries, all_cards):
		"""
		Transform queries to numpy array
		"""
		num_feats = 2 * self.num_cols
		num_queries = len(all_queries)
		X1 = np.zeros(shape=(num_queries, self.num_cols), dtype=np.float64)
		X2 = np.zeros(shape=(num_queries, self.num_cols), dtype=np.float64) + 1000
		X = np.hstack((X1, X2))
		print("X shape: ", X.shape)
		Y = np.reshape(np.array(all_cards), newshape=(num_queries, 1))
		for i, pred_list in enumerate(all_queries):
			for (col_idx, upper, lower) in pred_list:
				# attribute normalization
				upper = (upper - self.all_col_ranges[col_idx][0]) / (self.all_col_ranges[col_idx][1] - self.all_col_ranges[col_idx][0]) * 1000
				lower = (lower - self.all_col_ranges[col_idx][0]) / (
							self.all_col_ranges[col_idx][1] - self.all_col_ranges[col_idx][0]) * 1000
				X[i][ col_idx ] = upper
				X[i][self.num_cols + col_idx] = lower
		Y = np.log2(Y)
		return X, Y



