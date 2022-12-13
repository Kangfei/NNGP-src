import os
import pandas as pd
import random
import numpy as np
import networkx as nx
import collections
import math

Address = collections.namedtuple('Address', ['start', 'end'])
QueryInfo = collections.namedtuple('QueryInfo', ['num_table', 'num_joins', 'num_predicates', 'is_equal_join', 'is_multi_key'])
JoinInfo = collections.namedtuple('JoinInfo', ['t1_id', 't2_id', 'col_name', 'col_type'])

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



class NNGPEncoder(object):
	def __init__(self, tables: list):
		self.tables = tables
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

	def parse_line_without_card_then_encode(self, line:str):
		terms = line.strip().split('@')
		table_str, join_str = terms[0].strip(), terms[-1].strip()
		table_names = table_str.split(',')
		table_ids = [ self.table_name_to_tid[table_name] for table_name in table_names]
		assert len(table_ids) + 2 == len(terms), "Query Format Error!"
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
		x = self.transform_to_1d_array(table_ids, all_pred_list, join_infos)
		return x


	def load_queries(self, query_path: str, use_aux: bool, q_error_threshold: float, coef_var_threshold:float):
		sub_dirs = os.listdir(query_path)
		all_queries, all_cards = [], []
		all_query_infos = []
		for sub_dir in sorted(sub_dirs):
			# load auxiliary queries
			if sub_dir == 'join_query_aux.txt':
				if not use_aux:
					continue
				with open(os.path.join(query_path, sub_dir), 'r') as in_file:
					for line in in_file:
						items = line.strip().split('@')
						q_error, coef_var = float(items[-2]), float(items[-1])
						#print(q_error, q_error_threshold)
						if q_error < q_error_threshold and coef_var < coef_var_threshold:
							continue
						line_str = '@'.join(items[: len(items) - 2])
						table_ids, all_pred_list, join_infos, card = self.parse_line(line_str)
						all_queries.append((table_ids, all_pred_list, join_infos))
						all_cards.append(card)
						table_pairs = set([(join_info.t1_id, join_info.t2_id) for join_info in join_infos])
						is_multi_key = True if len(table_pairs) < len(join_infos) else False
						num_predicates = sum([len(pred_list) for pred_list in all_pred_list])
						all_query_infos.append(QueryInfo(num_table=len(table_ids), num_joins=len(join_infos),
														 num_predicates=num_predicates, is_equal_join=True,
														 is_multi_key=is_multi_key))
					in_file.close()
				continue
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
