import os
import pandas as pd
import random
import numpy as np
import networkx as nx
import datasets
import collections
import math
from QuerySampler import Address, QueryInfo, JoinInfo
from pandasql import sqldf
from multiprocessing import Process
from util import make_dir, transform_category_encoding
import clickhouse_driver

def resolve_conflict_col_name(df_left, df_right, join_cols):
	left_names = list(df_left.names)
	right_names = list(df_right.names)
	conflict_names = list((set(left_names) & set(right_names)) - set(join_cols))
	left_names_dict = { name : "{}_x".format(name) for name in conflict_names }
	right_names_dict = { name : "{}_y".format(name) for name in conflict_names }
	df_left = df_left.rename(columns = left_names_dict)
	df_right = df_right.rename(colums = right_names_dict)
	return df_left, df_right

class Table(object):
	def __init__(self, df, col_types, table_name:str, chunk_size=10, fk_code_dicts=None):
		self.df = df
		self.table_name = table_name
		self.col_types = col_types

		self.num_cols = len(df.columns)
		self.num_rows = len(df.index)
		self.all_col_ranges = np.zeros(shape=(self.num_cols, 2))
		self.all_col_denominator = np.zeros(shape=(self.num_cols,))
		self.df.fillna(-1, inplace=True)
		self.all_col_df = []
		self.categorical_codes_dict = {}
		self.chunk_size = chunk_size
		self.all_col_address = []  # [Address] : the address ([start, end)) of encoding in the feature of column i
		self.table_feat_dim = 0

		for i in range(self.num_cols):
			col_name = self.df.columns[i]
			single_col_df = self.df.iloc[:, i]
			single_col_df = single_col_df.sort_values()
			self.all_col_df.append(single_col_df)
			if col_types[i] == 'categorical':
				# categorical type
				cate = pd.Categorical(single_col_df)
				#print(len(single_col_df.unique()))
				#print(type(self.df.columns[i]))
				if fk_code_dicts is not None and col_name in fk_code_dicts.keys():
					self.categorical_codes_dict[col_name] = fk_code_dicts[col_name]
				else:
					self.categorical_codes_dict[col_name] = \
						dict([(category, code) for code, category in enumerate(cate.categories)]) # {category : code}
				num_cat = len(self.categorical_codes_dict[col_name])
				encode_dim = math.ceil(float(num_cat) / self.chunk_size)
				print(self.table_name, col_name, num_cat, encode_dim)
				self.all_col_address.append(Address(start=self.table_feat_dim, end=self.table_feat_dim + encode_dim))
				self.table_feat_dim += encode_dim
			else: # numerical type
				self.all_col_ranges[i][0] = single_col_df.min()
				self.all_col_ranges[i][1] = single_col_df.max()
				denominator = self.all_col_ranges[i][1] - self.all_col_ranges[i][0]
				self.all_col_denominator[i] = denominator if denominator > 0 else 1e-6
				self.all_col_address.append(Address(start=self.table_feat_dim, end=self.table_feat_dim + 2))
				self.table_feat_dim += 2

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
		codes_dict = self.categorical_codes_dict[col_name] #{category : code}
		cat_size = min(cat_size, len(codes_dict))
		if data_centric:
			cat_set = df.iloc[random.sample(range(len(df.index)), cat_size)]
			cat_set = list(set(cat_set)) # remove duplicates cat value
		else:
			cat_set = random.sample(list(codes_dict.keys()), cat_size)

		predicate = [  '{} == {}'.format(col_name, str(cat_pred)) for cat_pred in cat_set ]
		predicate = ' (' + ' or '.join(predicate) + ') '
		cat_set = [str(codes_dict[cat]) for cat in cat_set]
		return predicate, col_name, cat_set

	def parse_predicates(self, pred_str: str):
		pred_list = []
		if not pred_str:
			return pred_list
		predicates = pred_str.split("#")
		for predicate in predicates:
			col_name = predicate.split(",")[0].strip()
			col_idx = self.df.columns.get_loc(col_name)
			if self.col_types[col_idx] == 'categorical':
				#print(col_name, self.table_name)
				cat_set = [ int(_.strip()) for _ in predicate.split(",")[1:]]
				pred_list.append((col_idx, cat_set))
			else: # numerical type
				upper, lower = float(predicate.split(",")[1].strip()), float(predicate.split(",")[2].strip())
				pred_list.append((col_idx, upper, lower))
		return pred_list

	def predicate_encoding(self, pred_list):
		# predicate encoding used for DNN
		x = np.zeros(shape=(self.table_feat_dim,), dtype=np.float64)
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
				upper = (upper - self.all_col_ranges[col_idx][0]) / self.all_col_denominator[col_idx] * 1000
				lower = (lower - self.all_col_ranges[col_idx][0]) / self.all_col_denominator[col_idx] * 1000
				x[encode_address.start] =  upper
				x[encode_address.start + 1] = lower
		return x

	def _factorized_encoding(self, col_idx, cat_set):
		assert self.col_types[col_idx] == 'categorical', 'Only categorical attribute supports factorized encoding！'
		encode_address = self.all_col_address[col_idx]
		encode_dim = encode_address.end - encode_address.start
		encoding_str = ['0'] * (encode_dim * self.chunk_size)
		#print(encode_dim, self.chunk_size)
		cat_set = [int(cat) for cat in cat_set]
		for cat in cat_set:
			#print(cat)
			encoding_str[cat] = '1'
		encoding_str = "".join(encoding_str)
		encoding_str = [encoding_str[ i : i + self.chunk_size] for i in range(0, len(encoding_str), self.chunk_size)]
		factorized_encoding = [int(code, 2) for code in encoding_str]
		return factorized_encoding

	def one_hot_predicate_encoding(self, pred_list):
		# predicate encoding for MSCN, TreeLSTM
		# only support numerical attribute
		cols_x = np.zeros(shape=(2 * len(pred_list), self.num_cols)) #
		ops_x = np.zeros(shape=(2 * len(pred_list), 3)) # 2 predicate_ops columns and 1 val column
		for i, pred in enumerate(pred_list):
			col_idx = pred[0]
			if self.col_types[col_idx] == 'numerical':
				upper, lower = pred[1], pred[2]
				upper = (upper - self.all_col_ranges[col_idx][0]) / self.all_col_denominator[col_idx] * 1000
				lower = (lower - self.all_col_ranges[col_idx][0]) / self.all_col_denominator[col_idx] * 1000
				cols_x[2 * i, col_idx] = 1
				ops_x[2 * i, 0] = 1
				ops_x[2 * i, 2] = upper

				cols_x[2 * i + 1, col_idx] = 1
				ops_x[2 * i + 1, 1] = 1
				ops_x[2 * i + 1, 2] = lower
			else:
				assert False, "To do ..."
		return cols_x, ops_x


	def query_pred(self, full_pred):
		return self.df.query(full_pred, engine='python')



class BinaryJoinQuerySampler(object):
	def __init__(self, table1: Table, table2: Table):
		self.table1 = table1
		self.table2 = table2
		self.df1, self.df2 = self.table1.df, self.table2.df
		self.join_col_names, self.join_col_types = [], []
		for col_name in self.df1.columns:
			if col_name in self.df2.columns and \
				self.table1.col_types[self.df1.columns.get_loc(col_name)] == self.table2.col_types[self.df2.columns.get_loc(col_name)]:
				self.join_col_names.append(col_name)
				self.join_col_types.append(self.table1.col_types[self.df1.columns.get_loc(col_name)])
		#### support join operator ##
		self.numerical_join_ops = ['<', '>', '=', '<=', '>=', '<>']
		self.categorical_join_ops = ['=', '<>']
		self.total_num_joins = len(self.join_col_names)
		self.join_ops_dict = {'>' : 0, '<' : 1, '=' : 2}
		self.join_feat_dim = self.total_num_joins * len(self.join_ops_dict)
		print("join feat dim = {}".format(self.join_feat_dim))



	def sample_join_query(self, num_joins, data_centric=False, cat_size = 10):
		assert 1 <= num_joins <= self.total_num_joins, "Error number of joins!"
		join_col_indices = random.sample(range(self.total_num_joins), k=num_joins)
		join_conditions = []
		join_cols = []
		for join_col_idx in join_col_indices:
			op = random.choice(self.categorical_join_ops) \
				if self.join_col_types[join_col_idx] == 'categorical' \
				else random.choice(self.numerical_join_ops)
			join_cols.append(self.join_col_names[join_col_idx])
			join_conditions.append((self.join_col_names[join_col_idx], op))

		t1_col_indices = [self.df1.columns.get_loc(col_name) for col_name in self.df1.columns if col_name not in join_cols]
		t2_col_indices = [self.df2.columns.get_loc(col_name) for col_name in self.df2.columns if col_name not in join_cols]
		t1_full_pred, t1_pred_str = self.sample_pred_query(self.table1, t1_col_indices, data_centric, cat_size)
		t2_full_pred, t2_pred_str = self.sample_pred_query(self.table2, t2_col_indices, data_centric, cat_size)
		full_join = [ 't1.{} {} t2.{}'.format(col_name, op, col_name) for (col_name, op) in join_conditions]
		full_join = ' and '.join(full_join)
		join_str = [','.join([col_name, op]) for (col_name, op) in join_conditions]
		join_str = '#'.join(join_str)
		#print(full_join)
		#print(t1_full_pred)
		#print(t2_full_pred)
		return t1_full_pred, t2_full_pred, full_join, t1_pred_str, t2_pred_str, join_str

	def query_true_card(self, t1_full_pred, t2_full_pred, full_join):
		t1 = self.table1.query_pred(t1_full_pred) if t1_full_pred.strip() else self.df1
		t2 = self.table2.query_pred(t2_full_pred) if t2_full_pred.strip() else self.df2
		# print(t1.shape, t2.shape)

		cond_join = 'select count(*) from t2 join t1 on ' + full_join + ';'
		# print(cond_join)
		card = sqldf(cond_join, locals()).iloc[0, 0]
		return card


	def sample_pred_query(self, table: Table, t_col_indices, data_centric= False, cat_size=10):
		full_pred, pred_str = [], []
		d = random.choice(range(len(t_col_indices) + 1))
		col_indices = random.sample(t_col_indices, k=d)
		col_indices.sort()
		for col_idx in col_indices:
			if table.col_types[col_idx] == 'categorical':
				predicate, col_name, cat_set = table.sample_categorical_col_predicate(col_idx, data_centric, cat_size)
				pred_str.append(",".join([col_name] + cat_set))
			else:
				predicate, col_name, upper, lower = table.sample_numeric_col_predicate(col_idx, data_centric)
				pred_str.append(",".join([col_name, str(upper), str(lower)]))
			full_pred.append(predicate)
		# merge all predicates for all columns
		full_pred = " and ".join(full_pred) # used for query the true card
		pred_str = "#".join(pred_str) # used for encoding
		return full_pred, pred_str

	def join_encoding(self, join_conditions):
		join_x = np.zeros((self.join_feat_dim,), dtype=np.float64)
		for (col_name, op) in join_conditions:
			for c in op:
				idx = self.join_col_names.index(col_name)
				join_x[ idx * len(self.join_ops_dict) + self.join_ops_dict[c]] = 1
		return join_x

	def transform_to_1d_array(self, t1_pred_list, t2_pred_list, join_conditions):
		t1_pred_x = self.table1.predicate_encoding(t1_pred_list)
		t2_pred_x = self.table2.predicate_encoding(t2_pred_list)
		join_x = self.join_encoding(join_conditions)
		x = np.hstack([t1_pred_x, t2_pred_x, join_x])
		return x

	def parse_line(self, line:str):
		terms = line.strip().split('@')
		t1_pred_str, t2_pred_str, join_str, card = terms[0].strip(), terms[1].strip(), terms[2].strip(), int(terms[3].strip())
		t1_pred_list = self.table1.parse_predicates(t1_pred_str)
		t2_pred_list = self.table2.parse_predicates(t2_pred_str)
		join_str = join_str.split('#')
		join_conditions = [(join.split(',')[0].strip(), join.split(',')[1].strip()) for join in join_str]
		#print("t1", t1_pred_str)
		#print("t2", t2_pred_str)
		#self.transform_to_1d_array(t1_pred_list, t2_pred_list, join_conditions)
		return t1_pred_list, t2_pred_list, join_conditions, card

	def load_queries(self, query_path):
		sub_dirs = os.listdir(query_path)
		all_queries, all_cards = [], []
		for sub_dir in sorted(sub_dirs):
			with open(os.path.join(query_path, sub_dir), "r") as in_file:
				for line in in_file:
					#print(line)
					t1_pred_list, t2_pred_list, join_conditions, card = self.parse_line(line)
					all_queries.append((t1_pred_list, t2_pred_list, join_conditions))
					all_cards.append(card)
				in_file.close()
		all_query_infos = self.analyze_queries(all_queries)
		return all_queries, all_cards, all_query_infos

	def analyze_queries(self, all_queries):
		all_query_infos = []
		for (t1_pred_list, t2_pred_list, join_conditions) in all_queries:
			is_multi_key = True if len(join_conditions) > 1 else False
			is_equal_join = True
			for (_, op) in join_conditions:
				if op != "=":
					is_equal_join = False
			all_query_infos.append(QueryInfo(num_table=2, num_joins=len(join_conditions),
											num_predicates=len(t1_pred_list) + len(t2_pred_list),
											is_equal_join=is_equal_join, is_multi_key=is_multi_key))
		return all_query_infos

	def transform_to_arrays(self, all_queries, all_cards):
		# Transform queries to numpy array
		num_queries = len(all_queries)
		X = []
		for (t1_pred_list, t2_pred_list, join_conditions) in all_queries:
			X.append(self.transform_to_1d_array(t1_pred_list, t2_pred_list, join_conditions))
		X = np.array(X)
		Y = np.reshape(np.array(all_cards), newshape=(num_queries, 1))
		Y = np.log2(Y)
		return X, Y

	def test_encoding(self, num_joins):
		t1_full_pred, t2_full_pred, full_join, t1_pred_str, t2_pred_str, join_str = self.sample_join_query(num_joins)
		card = self.query_true_card(t1_full_pred, t2_full_pred, full_join)
		query_str = t1_pred_str + '@' + t2_pred_str + '@' + join_str + '@' + str(card) + '\n'
		print(query_str)
		t1_pred_list, t2_pred_list, join_conditions, card = self.parse_line(query_str)
		print(t1_pred_list)
		print(t2_pred_list)
		print(join_conditions)
		x = self.transform_to_1d_array(t1_pred_list, t2_pred_list, join_conditions)
		print(x.shape)
		print(x)


	def sample_batch_query(self, num_joins, mini_batch, cat_size):
		i = 0
		query_card = {}
		save_path = "./queryset/join_{}_{}_{}_2" \
			.format(self.table1.table_name, self.table2.table_name, cat_size)
		make_dir(save_path)
		with open(os.path.join(save_path, "join_query_{}.txt".format(num_joins)), 'a') as in_file:
			while i < mini_batch:
				t1_full_pred, t2_full_pred, full_join, t1_pred_str, t2_pred_str, join_str =\
					self.sample_join_query(num_joins, data_centric=True, cat_size=cat_size)
				query_str = t1_pred_str + '@' + t2_pred_str + '@' + join_str
				if query_str in query_card.keys():
					continue
				card = self.query_true_card(t1_full_pred, t2_full_pred, full_join)
				# sample a unique query
				query_card[query_str] = card
				if card < 1:
					continue
				print(query_str + "@" + str(card))
				# save sampled queries
				in_file.write(query_str + "@" + str(card) + "\n")
				i += 1
		in_file.close()


	def parallel_sampler(self, mini_batch, cat_size=50):
		for num_joins in range(1, self.total_num_joins):
			p = Process(target=self.sample_batch_query,args=(num_joins, mini_batch, cat_size))
			p.start()


class MultiJoinQuerySampler(object):
	def __init__(self, tables: list, query_engine="pandas"):
		self.tables = tables
		self.query_engine = query_engine
		self.num_tables = len(tables)
		self.tid_to_table_name, self.table_name_to_tid = {}, {}
		for i, table in enumerate(tables):
			self.tid_to_table_name[i] = table.table_name
			self.table_name_to_tid[table.table_name] = i
		self.schema_name = "_".join([table.table_name for table in tables])
		self.all_join_infos = []
		self.tid_to_join_infos = {}
		self.table_pair_to_join_infos = {}
		for t1_id in range(self.num_tables - 1):
			table1 = self.tables[t1_id]
			df1 = table1.df
			for t2_id in range(t1_id + 1, self.num_tables):
				table2 = self.tables[t2_id]
				df2 = table2.df
				for col_name in df1.columns:
					if col_name in df2.columns and \
							table1.col_types[df1.columns.get_loc(col_name)] == table2.col_types[
						df2.columns.get_loc(col_name)]:
						join_info = JoinInfo(t1_id=t1_id, t2_id=t2_id, col_name=col_name, col_type=table1.col_types[df1.columns.get_loc(col_name)])
						self.all_join_infos.append(join_info)
						if t1_id not in self.tid_to_join_infos.keys():
							self.tid_to_join_infos[t1_id] = []
						self.tid_to_join_infos[t1_id].append(join_info)
						if t2_id not in self.tid_to_join_infos.keys():
							self.tid_to_join_infos[t2_id] = []
						self.tid_to_join_infos[t2_id].append(join_info)
						if (t1_id, t2_id) not in self.table_pair_to_join_infos.keys():
							self.table_pair_to_join_infos[(t1_id, t2_id)] = []
						self.table_pair_to_join_infos[(t1_id, t2_id)].append(join_info)
		self.all_join_table_pairs = list(self.table_pair_to_join_infos.keys())
		self.join_graph = nx.Graph()
		self.join_graph.add_edges_from(self.all_join_table_pairs)
		self.all_join_triples = [(join_info.t1_id, join_info.t2_id, join_info.col_name) for join_info in self.all_join_infos]
		self.all_join_col_names = [join_info.col_name for join_info in self.all_join_infos]
		#### support join operator ##
		self.numerical_join_ops = ['<', '>', '=', '<=', '>=', '<>']
		self.categorical_join_ops = ['=', '<>']
		self.total_num_joins = len(self.all_join_triples)
		self.join_ops_dict = {'>': 0, '<': 1, '=': 2}
		self.join_feat_dim = self.total_num_joins * len(self.join_ops_dict)

		print("join feat dim = {}".format(self.join_feat_dim))


	def sample_tables_and_joins(self, num_tables):
		table_ids, join_infos = [],[]

		sample_frontier = set()
		start = random.choice(range(self.num_tables))
		#start = 0
		table_ids.append(start)
		for neighbor in self.join_graph.adj[start]:
			candidate_pair = (start, neighbor) if start < neighbor else (neighbor, start)
			sample_frontier.add(candidate_pair)
		while len(table_ids) < num_tables and len(sample_frontier) > 0:
			(t1_id, t2_id) = sample_frontier.pop()
			if t1_id in table_ids and t2_id in table_ids:
				continue
			cur_tid = t1_id if t2_id in table_ids else t2_id
			table_ids.append(cur_tid)
			# sample one joins from t1-t2
			sampled_join = random.choice(self.table_pair_to_join_infos[(t1_id, t2_id)])
			join_infos.append(sampled_join)
			for next_tid in self.join_graph.adj[cur_tid]:
				if next_tid in table_ids: # avoid cyclic join
					continue
				candidate_pair = (cur_tid, next_tid) if cur_tid < next_tid else (next_tid, cur_tid)
				sample_frontier.add(candidate_pair)
		return sorted(table_ids), join_infos

	def sample_pred_query(self, table: Table, t_col_indices, data_centric= False, cat_size=10):
		full_pred, pred_str = [], []
		d = random.choice(range(int(len(t_col_indices)) + 1))
		#d = 1
		col_indices = random.sample(t_col_indices, k=d)
		col_indices.sort()
		for col_idx in col_indices:
			if table.col_types[col_idx] == 'categorical':
				predicate, col_name, cat_set = table.sample_categorical_col_predicate(col_idx, data_centric, cat_size)
				pred_str.append(",".join([col_name] + cat_set))
			else:
				predicate, col_name, upper, lower = table.sample_numeric_col_predicate(col_idx, data_centric)
				pred_str.append(",".join([col_name, str(upper), str(lower)]))
			full_pred.append(predicate)
		# merge all predicates for all columns
		full_pred = " and ".join(full_pred) # used for query the true card
		pred_str = "#".join(pred_str) # used for encoding
		return full_pred, pred_str

	def sample_join_query(self, num_tables, data_centric=False, cat_size = 10):
		if num_tables == 1:
			table_ids, join_infos = [random.choice(range(self.num_tables))], []
		else:
			table_ids, join_infos = self.sample_tables_and_joins(num_tables)
		join_cols = [join_info.col_name for join_info in join_infos]
		full_pred_list, pred_str_list = [], []
		for t_id in table_ids:
			table = self.tables[t_id]
			#t_col_indices = [table.df.columns.get_loc(col_name) for col_name in table.df.columns if col_name not in join_cols] #
			t_col_indices = [table.df.columns.get_loc(col_name) for col_name in table.df.columns if
							 col_name not in join_cols and col_name not in self.all_join_col_names] # avoid to sample range query over join keys
			full_pred, pred_str = self.sample_pred_query(table, t_col_indices, data_centric, cat_size)
			full_pred_list.append(full_pred)
			pred_str_list.append(pred_str)
		join_str = [",".join([self.tid_to_table_name[join_info.t1_id], self.tid_to_table_name[join_info.t2_id], join_info.col_name])
					for join_info in join_infos]
		join_str = '#'.join(join_str)
		return table_ids, full_pred_list , pred_str_list, join_infos, join_str

	def query_true_card(self, table_ids, full_pred_list, join_infos):
		full_join = ['{}.{} = {}.{}'.format(self._get_temp_table_name(join_info.t1_id), join_info.col_name,
										   self._get_temp_table_name(join_info.t2_id), join_info.col_name)
					for join_info in join_infos]
		full_join = ' and '.join(full_join)
		tmp_tables = locals()
		for t_id, full_pred in zip(table_ids, full_pred_list):
			table = self.tables[t_id]
			tmp_table_name = self._get_temp_table_name(t_id)
			tmp_tables[tmp_table_name] = table.query_pred(full_pred) if full_pred.strip() else table.df
			# early stop when one tmp_table is empty
			if len(tmp_tables[tmp_table_name].index) == 0:
				return 0

		tables_in_FROM = [self._get_temp_table_name(t_id) for t_id in table_ids]
		tables_in_FROM = ",".join(tables_in_FROM)
		conj_join = 'select count(*) from ' + tables_in_FROM + ' where ' + full_join + ';' if len(join_infos) > 0 else \
			'select count(*) from ' + tables_in_FROM + ';'
		card = sqldf(conj_join, locals()).iloc[0, 0]
		return card

	def query_true_card_by_clickhouse(self, temp_query_str):
		def Div(s):
			table_name_list = []
			preds_list = []
			join_conditions = []
			card = 0
			# Line := table_name_list + ‘@’ + preds_list + ‘@’ + join_conditions + ‘@’ + card
			temp = s.split('@')
			# table_name_list: = table_name | [table_name + ‘,’]
			table_name_list = temp[0].split(',')
			# preds_list := preds | [ preds + ‘@’]
			# preds := numerical_pred | [numerical_pred + ‘#’]
			# numerical_pred := column_name + ‘,’ + upper_value + ‘,’  + lower_value
			preds_list = temp[1:-1]
			for i in range(0, len(preds_list)):
				preds_list[i] = preds_list[i].split('#')
				for j in range(0, len(preds_list[i])):
					preds_list[i][j] = preds_list[i][j].split(',')
				# preds_list[i][j][0] = preds_list[i][j][0].replace(table_name_list[i] + '_', '', 1)
			# join_conditions = join_condition | [join_condition + ‘#’]
			# join_condition := table1_name + ’,’ +  table2_name + ’,’ + join_column_name
			#                 | table1_name + ’,’ +  table2_name + ’,’ + join_column_name + ‘,’ + join_operator (useless)
			join_conditions = temp[-1].split('#')
			for i in range(0, len(join_conditions)):
				join_conditions[i] = join_conditions[i].split(',')
			return table_name_list, preds_list, join_conditions

		def ToSQL(table_name_list, preds_list, join_conditions):
			HEAD = "SELECT COUNT(*) FROM "
			# table_name_list = ','.join(table_name_list)
			WHERE = " WHERE "
			tmp = []
			for i in range(0, len(table_name_list)):
				for j in range(0, len(preds_list[i])):
					if (len(preds_list[i][j]) != 3):
						continue
					tmp.append(table_name_list[i] + '.' + preds_list[i][j][0] + ' <= ' + preds_list[i][j][1])
					tmp.append(table_name_list[i] + '.' + preds_list[i][j][0] + ' >= ' + preds_list[i][j][2])
			for i in range(0, len(join_conditions)):
				if (len(join_conditions[i]) != 3):
					continue
				temp = join_conditions[i][0] + '.' + join_conditions[i][2] + '=' + join_conditions[i][1] + '.' + \
					   join_conditions[i][2]
				tmp.append(temp)
			table_name_list = ','.join(table_name_list)
			conditions = ' AND '.join(tmp)
			res = HEAD + table_name_list
			if (len(tmp) > 0):
				res = res + WHERE + conditions
			#res = res + ';'
			return res
		print(temp_query_str)
		table_name_list, preds_list, join_conditions = Div(temp_query_str)
		print(table_name_list)
		print(preds_list)
		print(join_conditions)
		sql_str = ToSQL(table_name_list, preds_list, join_conditions)
		client = clickhouse_driver.Client(host='localhost', port='9000', database='imdb')
		res = client.execute(sql_str)
		card = res[0][0]
		return card

	def sample_batch_query(self, save_path, num_tables, mini_batch, data_centric=False, cat_size=10):
		i = 0
		query_card = {}

		with open(os.path.join(save_path, "join_query_{}.txt".format(num_tables)), 'a') as in_file:
			while i < mini_batch:
				table_ids, full_pred_list, pred_str_list, join_infos, join_str = \
					self.sample_join_query(num_tables, data_centric, cat_size)

				tables_names = [self.tid_to_table_name[t_id] for t_id in table_ids]
				tables_names = ",".join(tables_names)
				pred_str = "@".join(pred_str_list)
				query_str = tables_names + "@" + pred_str + "@" + join_str
				# sample a unique query
				if query_str in query_card.keys():
					continue

				if self.query_engine == 'pandas':
					card = self.query_true_card(table_ids, full_pred_list, join_infos)
				else:
					card = self.query_true_card_by_clickhouse(query_str)
				if card < 1:
					continue
				print(query_str + "@" + str(card))
				# save sampled queries
				in_file.write(query_str + "@" + str(card) + "\n")
				i += 1
		in_file.close()

	def parallel_sampler(self, mini_batch, data_centric= False, cat_size=10):
		save_path = "./queryset/join_{}_{}".format(self.schema_name, cat_size) if not data_centric else \
			"./queryset/join_{}_{}_data_centric_824_FP".format(self.schema_name, cat_size)
		make_dir(save_path)
		for num_tables in range(1, self.num_tables + 1):
			p = Process(target=self.sample_batch_query, args=(save_path, num_tables, mini_batch, data_centric, cat_size))
			p.start()

	def join_encoding(self, join_infos):
		join_x  = np.zeros((self.join_feat_dim,), dtype=np.float64)
		for join_info in join_infos:
			t1_id, t2_id, col_name, op = join_info.t1_id, join_info.t2_id, join_info.col_name, "="
			join_triple = (t1_id, t2_id, col_name) if t1_id < t2_id else (t2_id, t1_id, col_name)
			idx = self.all_join_triples.index(join_triple)
			for c in op:
				join_x[idx * len(self.join_ops_dict) + self.join_ops_dict[c]] = 1
		return join_x

	def transform_to_1d_array(self, table_ids, all_pred_list, join_infos):
		encodings = []
		for t_id in range(self.num_tables):
			pred_list = all_pred_list[table_ids.index(t_id)] if t_id in table_ids else []
			encode = self.tables[t_id].predicate_encoding(pred_list)
			encodings.append(encode)
		encodings.append(self.join_encoding(join_infos))
		x = np.hstack(encodings)
		return x

	def parse_line(self, line:str):
		terms = line.strip().split('@')
		table_str, join_str, card = terms[0].strip(), terms[-2].strip(), int(terms[-1].strip())
		table_names = table_str.split(',')
		table_ids = [ self.table_name_to_tid[table_name] for table_name in table_names]
		assert len(table_ids) + 3 == len(terms), "Query Format Error!"
		all_pred_str = terms[1 : len(table_ids) + 1]
		all_pred_list, join_infos = [], []
		for t_id, pred_str in zip(table_ids, all_pred_str):
			pred_list = self.tables[t_id].parse_predicates(pred_str.strip())
			all_pred_list.append(pred_list)
		join_str = [] if not join_str else join_str.split('#')
		#print(join_str)
		for join in join_str:
			t1_name, t2_name, col_name = join.split(',')[0].strip(), join.split(',')[1].strip(), join.split(',')[2].strip()
			t_id = self.table_name_to_tid[t1_name]
			col_idx = self.tables[t_id].df.columns.get_loc(col_name)
			col_type = self.tables[t_id].col_types[col_idx]
			join_info = JoinInfo(t1_id=self.table_name_to_tid[t1_name], t2_id=self.table_name_to_tid[t2_name], col_name=col_name, col_type=col_type)
			join_infos.append(join_info)
		return table_ids, all_pred_list, join_infos, card


	def load_queries(self, query_path):
		sub_dirs = os.listdir(query_path)
		all_queries, all_cards = [], []
		all_query_infos = []
		for sub_dir in sorted(sub_dirs):
			with open(os.path.join(query_path, sub_dir), "r") as in_file:
				for line in in_file:
					table_ids, all_pred_list, join_infos, card = self.parse_line(line)
					all_queries.append((table_ids, all_pred_list, join_infos))
					all_cards.append(card)
					table_pairs = set([(join_info.t1_id, join_info.t2_id) for join_info in join_infos])
					is_multi_key = True if len(table_pairs) < len(join_infos) else False
					num_predicates = sum([len(pred_list) for pred_list in all_pred_list])
					all_query_infos.append(QueryInfo(num_table=len(table_ids), num_joins=len(join_infos),
													 num_predicates=num_predicates, is_equal_join=True,
													 is_multi_key=is_multi_key))
				in_file.close()
		return all_queries, all_cards, all_query_infos


	def transform_to_arrays(self, all_queries, all_cards):
		# Transform queries to numpy array
		num_queries = len(all_queries)
		X = []
		for (table_ids, all_pred_list, join_infos) in all_queries:
			X.append(self.transform_to_1d_array(table_ids, all_pred_list, join_infos))
		X = np.array(X)
		Y = np.reshape(np.array(all_cards), newshape=(num_queries, 1))
		Y = np.log2(Y)
		return X, Y

	def _get_temp_table_name(self, t_id):
		if t_id in self.tid_to_table_name.keys():
			return self.tid_to_table_name[t_id] + '_tmp'
		elif t_id in self.table_name_to_tid.keys():
			return t_id + '_tmp'
		else:
			assert False, "table id not applicable."


if __name__ == "__main__":

	"""
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
	"""


	df_dataset1, col_types1 = datasets.LoadYelp_Reviews(nrows=100000)
	"""
	df_dataset1 = transform_category_encoding(df_dataset1, col_types1)
	csv_file = "/home/kfzhao/data/rdb/yelp/user_tmp.csv"
	df_dataset1.to_csv(path_or_buf=csv_file, sep=';')
	"""
	df_dataset2, col_types2 = datasets.LoadYelp_Users(nrows=100000)
	table1 = Table(df_dataset1, col_types1, 'review')
	table2 = Table(df_dataset2, col_types2, 'user')

	query_sampler = BinaryJoinQuerySampler(table1, table2)
	#query_sampler.parallel_sampler(mini_batch=4000, cat_size=100)
	#query_sampler.test_encoding(num_joins=3)
	all_queries, all_cards, all_query_infos = query_sampler.load_queries(query_path='/home/kfzhao/PycharmProjects/NNGP/queryset/join_review_user_100_2')
	X, Y = query_sampler.transform_to_arrays(all_queries, all_cards)
	print(X.shape, Y.shape)


