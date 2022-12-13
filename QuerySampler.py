import os
import pandas as pd
import random
import numpy as np
import datasets
import collections
import math
from multiprocessing import Process
from util import make_dir

Address = collections.namedtuple('Address', ['start', 'end'])
QueryInfo = collections.namedtuple('QueryInfo', ['num_table', 'num_joins', 'num_predicates', 'is_equal_join', 'is_multi_key'])
JoinInfo = collections.namedtuple('JoinInfo', ['t1_id', 't2_id', 'col_name', 'col_type'])

class GeneralQuerySampler(object):
	def __init__(self, df, col_types, dataset, chunk_size=10):
		self.df = df
		self.col_types = col_types
		self.dataset = dataset

		self.num_cols = len(df.columns)
		self.num_rows = len(df.index)

		self.all_col_ranges = np.zeros(shape=(self.num_cols, 2))
		self.df.fillna(-1, inplace=True)
		self.all_col_df = []
		self.categorical_codes_dict = {}
		# for feature encoding
		self.total_feat_dim = 0
		self.chunk_size = chunk_size
		self.all_col_address = [] # [Address] : the address ([start, end)) of encoding in the feature of column i

		for i in range(self.num_cols):
			single_col_df = self.df.iloc[:, i]
			single_col_df = single_col_df.sort_values()
			self.all_col_df.append(single_col_df)
			if col_types[i] == 'categorical':
				cate = pd.Categorical(single_col_df)
				#print(type(self.df.columns[i]))
				self.categorical_codes_dict[self.df.columns[i]] = \
					dict([(category, code) for code, category in enumerate(cate.categories)]) # {category : code}
				num_cat = len(single_col_df.unique())
				encode_dim =  math.ceil(float(num_cat) / self.chunk_size)
				self.all_col_address.append(Address(start=self.total_feat_dim, end=self.total_feat_dim + encode_dim))
				self.total_feat_dim += encode_dim

				#print(num_cat, encode_dim)
				#print(cate.categories, len(cate.categories))
			else: # numerical type
				self.all_col_ranges[i][0] = single_col_df.min()
				self.all_col_ranges[i][1] = single_col_df.max()
				self.all_col_address.append(Address(start=self.total_feat_dim, end=self.total_feat_dim + 2))
				self.total_feat_dim += 2

		#print(self.categorical_codes_dict)
		print("feature dim={}".format(self.total_feat_dim))
		random.seed(1)

	def sample_numeric_col_predicate(self, col_idx, data_centric = False):
		df = self.all_col_df[col_idx]
		col_name = self.df.columns[col_idx]
		min_val, max_val = self.all_col_ranges[col_idx][0], self.all_col_ranges[col_idx][1]
		if data_centric:
			# Data centric sample
			val1 = df.iloc[random.randrange(0, len(df.index))]
			val2 = df.iloc[random.randrange(0, len(df.index))]
		else:
			# Random centric sample
			val1 = random.uniform(min_val, max_val)
			val2 = random.uniform(min_val, max_val)
		(upper, lower) = (val1, val2) if val1 >= val2 else (val2, val1)
		predicate = "{} <= {} and {} >= {}".format(col_name, upper, col_name, lower)

		return predicate, col_name, upper, lower

	def sample_categorical_col_predicate(self, col_idx, data_centric = False, cat_size= 1):
		df = self.all_col_df[col_idx]
		col_name = self.df.columns[col_idx]
		codes_dict = self.categorical_codes_dict[col_name]
		cat_size = min(cat_size, len(codes_dict))
		if data_centric:
			cat_set = df.iloc[random.sample(range(len(df.index)), cat_size)]
		else:
			cat_set = random.sample(list(codes_dict.keys()), cat_size)

		predicate = [  '{} == {}'.format(col_name, str(cat_pred)) for cat_pred in cat_set ]
		predicate = ' (' + ' or '.join(predicate) + ') '
		cat_set = [str(codes_dict[cat]) for cat in cat_set]
		return predicate, col_name, cat_set


	def sample_query(self, d, data_centric= False, cat_size= 1):
		# sample a range query with d columns
		assert 0 < d <= self.num_cols, "Error Attribute Number to Sample!"
		full_pred,  pred_str = [],[]
		col_indices = random.sample(range(self.num_cols), k=d)
		col_indices.sort()
		for col_idx in col_indices:
			if self.col_types[col_idx] == 'categorical':
				predicate, col_name, cat_set = self.sample_categorical_col_predicate(col_idx, data_centric, cat_size)
				pred_str.append(",".join([col_name] + cat_set))
			else:
				predicate, col_name, upper, lower = self.sample_numeric_col_predicate(col_idx, data_centric)
				pred_str.append(",".join([col_name, str(upper), str(lower)]))
			full_pred.append(predicate)
		# merge all predicates
		full_pred = " and ".join(full_pred)
		pred_str = "#".join(pred_str)
		return full_pred, pred_str

	def query_true_card(self, full_pred):
		return len(self.df.query(full_pred, engine='python').index)

	def sample_batch_query(self, d, mini_batch, cat_size=10):
		i = 0
		query_card = {}
		save_path = "./queryset/{}_{}".format(self.dataset, cat_size)
		make_dir(save_path)
		with open(os.path.join(save_path, "query_{}.txt".format(d)), 'a') as in_file:
			while i < mini_batch:
				full_pred, pred_str = self.sample_query(d, data_centric=False, cat_size=cat_size)
				if pred_str in query_card.keys():
					continue
				card = self.query_true_card(full_pred)
				# sample a unique query
				query_card[pred_str] = card
				if card < 1:
					continue
				print(pred_str + "@" + str(card))
				# save sampled queries
				in_file.write(pred_str + "@" + str(card) + "\n")
				i += 1
		in_file.close()

	def parallel_sample(self, mini_batch, cat_size=50):
		for d in range(1, self.num_cols + 1):
			p = Process(target=self.sample_batch_query, args=(d, mini_batch, cat_size))
			p.start()


	def test_encoding(self, mini_batch=20):
		for d in range(2, self.num_cols + 1):
			i = 0
			while i < mini_batch:
				full_pred, pred_str = self.sample_query(d)
				card = self.query_true_card(full_pred)
				output_str = pred_str + "@" + str(card) + "\n"
				pred_list, card = self.parse_line(output_str)
				print(pred_list, card)
				x = self.transform_to_1d_array(pred_list)
				print(np.sum(np.where(x > 0, 1, x)))
				print(x.shape)
				i += 1



	def parse_line(self, line: str):
		pred_str, card = line.split("@")[0].strip(), int(line.split("@")[1].strip())
		predicates = pred_str.split("#")
		pred_list = []
		for predicate in predicates:
			col_name = predicate.split(",")[0].strip()
			col_idx = self.df.columns.get_loc(col_name)
			if self.col_types[col_idx] == 'categorical':
				cat_set = [ int(_.strip()) for _ in predicate.split(",")[1:]]
				pred_list.append((col_idx, cat_set))
			else: # numerical type
				upper, lower = float(predicate.split(",")[1].strip()), float(predicate.split(",")[2].strip())
				pred_list.append((col_idx, upper, lower))
		return pred_list, card

	def load_queries(self, query_path):
		sub_dirs = os.listdir(query_path)
		all_queries, all_cards = [], []
		all_query_infos = []
		for sub_dir in sorted(sub_dirs):
			print(sub_dir)
			with open(os.path.join(query_path, sub_dir), "r") as in_file:
				for line in in_file:
					#print(line)
					pred_list, card = self.parse_line(line)
					all_queries.append(pred_list)
					all_cards.append(card)
					all_query_infos.append(QueryInfo(num_table=1, num_joins=0, num_predicates=len(pred_list), is_equal_join=False, is_multi_key=False))
				in_file.close()
		return all_queries, all_cards, all_query_infos

	def transform_to_arrays(self, all_queries, all_cards):
		# Transform queries to numpy array
		num_queries = len(all_queries)
		X = []
		for pred_list in all_queries:
			X.append(self.transform_to_1d_array(pred_list))
		X = np.array(X)
		Y = np.reshape(np.array(all_cards), newshape=(num_queries, 1))
		Y = np.log2(Y)
		return X, Y


	def transform_to_1d_array(self, pred_list):
		x = np.zeros(shape=(self.total_feat_dim,), dtype=np.float64)
		for col_idx in range(self.num_cols):
			if self.col_types[col_idx] == 'numerical':
				x[self.all_col_address[col_idx].start + 1] = 1000
		for pred in pred_list:
			col_idx = pred[0]
			encode_address = self.all_col_address[col_idx]
			if self.col_types[col_idx] == 'categorical':
				factorized_encoding = self._factorized_encoding(col_idx, pred[1])
				idx = list(range(encode_address.start, encode_address.end))
				np.put(x, idx, factorized_encoding)

			else:
				upper, lower = pred[1], pred[2]
				upper = (upper - self.all_col_ranges[col_idx][0]) / (
							self.all_col_ranges[col_idx][1] - self.all_col_ranges[col_idx][0]) * 1000
				lower = (lower - self.all_col_ranges[col_idx][0]) / (
						self.all_col_ranges[col_idx][1] - self.all_col_ranges[col_idx][0]) * 1000
				x[encode_address.start] =  upper
				x[encode_address.start + 1] = lower
		return x


	def _factorized_encoding(self, col_idx, cat_set):
		assert self.col_types[col_idx] == 'categorical', 'Only categorical attribute supports factorized encoding！'
		encode_address = self.all_col_address[col_idx]
		encode_dim = encode_address.end - encode_address.start
		encoding_str = ['0'] * (encode_dim * self.chunk_size)
		cat_set = [int(cat) for cat in cat_set]
		for cat in cat_set:
			encoding_str[cat] = '1'
		encoding_str = "".join(encoding_str)
		encoding_str = [encoding_str[ i : i + self.chunk_size] for i in range(0, len(encoding_str), self.chunk_size)]
		factorized_encoding = [int(code, 2) for code in encoding_str]
		return factorized_encoding


if __name__ == "__main__":
	dataset = 'yelp-user'
	if dataset == 'forest':
		df_dataset, col_types = datasets.LoadForest()
	elif dataset == 'higgs':
		df_dataset, col_types = datasets.LoadHiggs()
	elif dataset == 'sales':
		df_dataset, col_types = datasets.LoadSales()
	elif dataset == 'yelp-review':
		df_dataset, col_types = datasets.LoadYelp_Reviews()
	elif dataset == 'yelp-user':
		df_dataset, col_types = datasets.LoadYelp_Users()

	print(df_dataset.shape)
	print(df_dataset.dtypes)
	query_sampler = GeneralQuerySampler(df_dataset, col_types, dataset)
	query_sampler.parallel_sample(mini_batch=2000, cat_size=100)
