import numpy as np
import os
import random
from JoinQuerySampler import Table
import networkx as nx
from QuerySampler import JoinInfo, QueryInfo, Address
import torch



class MSCNEncoder(object):
	def __init__(self, table: Table):
		self.table = table
		self.col_types = self.table.col_types
		self.df = self.table.df

	def transform_to_1d_array(self, pred_list):
		cols_x, ops_x = self.table.one_hot_predicate_encoding(pred_list)
		pred_x = np.hstack([cols_x, ops_x])
		return pred_x

	def transform_to_arrays(self, all_queries, all_cards):
		num_queries = len(all_queries)
		X = []
		for pred_list in all_queries:
			X.append(self.transform_to_1d_array(pred_list))
		Y = np.reshape(np.array(all_cards), newshape=(num_queries, 1))
		Y = np.log2(Y)
		return X, Y


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
			with open(os.path.join(query_path, sub_dir), "r") as in_file:
				for line in in_file:
					#print(line)
					pred_list, card = self.parse_line(line)
					all_queries.append(pred_list)
					all_cards.append(card)
					all_query_infos.append(QueryInfo(num_table=1, num_joins=0, num_predicates=len(pred_list), is_equal_join=False, is_multi_key=False))
				in_file.close()
		return all_queries, all_cards, all_query_infos


class MSCNJoinQueryEncoder(object):
	def __init__(self, table1: Table, table2 :Table):
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
		self.join_ops = ['<', '>', '=', '<=', '>=', '<>']
		self.total_num_joins = len(self.join_col_names)
		self.join_feat_dim = self.total_num_joins + len(self.join_ops)

	def one_hot_join_encoding(self, join_conditions):
		num_joins = len(join_conditions)
		join_x = np.zeros(shape=(num_joins, self.join_feat_dim))
		for i, (col_name, op) in enumerate(join_conditions):
			col_idx = self.join_col_names.index(col_name)
			op_idx = self.join_ops.index(op)
			join_x[i, col_idx] = 1
			join_x[i, self.total_num_joins + op_idx] = 1
		return join_x

	def transform_to_1d_array(self, t1_pred_list, t2_pred_list, join_conditions):
		t1_cols_x, t1_ops_x = self.table1.one_hot_predicate_encoding(t1_pred_list)
		t2_cols_x, t2_ops_x = self.table2.one_hot_predicate_encoding(t2_pred_list)
		t1_cols = np.zeros(shape=(2 * len(t2_pred_list), self.table1.num_cols))
		t2_cols = np.zeros(shape=(2 * len(t1_pred_list), self.table2.num_cols))

		t1_pred_x = np.hstack([t1_cols_x, t2_cols, t1_ops_x])
		t2_pred_x = np.hstack([t1_cols, t2_cols_x, t2_ops_x])
		join_x = self.one_hot_join_encoding(join_conditions)
		return t1_pred_x, t2_pred_x, join_x


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
			t1_pred_x, t2_pred_x, join_x = self.transform_to_1d_array(t1_pred_list, t2_pred_list, join_conditions)
			X.append((t1_pred_x, t2_pred_x, join_x))
		Y = np.reshape(np.array(all_cards), newshape=(num_queries, 1))
		Y = np.log2(Y)
		return X, Y


class MultiJoinQueryEncoder(object):
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
						join_info = JoinInfo(t1_id=t1_id, t2_id=t2_id, col_name=col_name,
											 col_type=table1.col_types[df1.columns.get_loc(col_name)])
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
		self.all_join_pairs = list(self.table_pair_to_join_infos.keys())
		self.join_graph = nx.Graph()
		self.join_graph.add_edges_from(self.all_join_pairs)
		self.all_join_triples = [(join_info.t1_id, join_info.t2_id, join_info.col_name) for join_info in
								 self.all_join_infos]
		#### support join operator ##
		self.join_ops = ['<', '>', '=', '<=', '>=', '<>']
		self.total_num_joins = len(self.all_join_triples)
		self.join_feat_dim = self.total_num_joins + len(self.join_ops)
		self.pred_feat_dim = 0
		self.all_pred_address = []  # [Address] : the address ([start, end)) of encoding in the feature of column i
		for table in self.tables:
			self.all_pred_address.append(Address(start=self.pred_feat_dim, end = self.pred_feat_dim + table.num_cols))
			self.pred_feat_dim += table.num_cols
		print("join feat dim = {}".format(self.join_feat_dim))
		print("pred feat dim = {}".format(self.pred_feat_dim))


	def one_hot_join_encoding(self, join_infos):
		num_joins = len(join_infos)
		join_x = np.zeros(shape=(num_joins, self.join_feat_dim))
		for i, join_info in enumerate(join_infos):
			t1_id, t2_id, col_name, op = join_info.t1_id, join_info.t2_id, join_info.col_name, "="
			join_triple = (t1_id, t2_id, col_name) if t1_id < t2_id else (t2_id, t1_id, col_name)
			idx = self.all_join_triples.index(join_triple)
			op_idx = self.join_ops.index(op)
			join_x[i, idx] = 1
			join_x[i, self.total_num_joins + op_idx] = 1
		return join_x

	def one_hot_table_encoding(self, table_ids):
		table_x = np.zeros(shape=(len(table_ids), self.num_tables))
		for i, t_id in enumerate(table_ids):
			table_x[i, t_id] = 1
		return table_x

	def one_table_pred_encoding(self, t_id, pred_list):
		if not pred_list:
			one_table_pred_x = np.zeros(shape=(1, self.pred_feat_dim + 3))
			return one_table_pred_x
		col_x, ops_x = self.tables[t_id].one_hot_predicate_encoding(pred_list)
		start, end = self.all_pred_address[t_id].start, self.all_pred_address[t_id].end
		one_table_pred_x = [np.zeros(shape=(col_x.shape[0], start), ), col_x,
							np.zeros(shape=(col_x.shape[0], self.pred_feat_dim - end), ), ops_x]
		one_table_pred_x = np.hstack(one_table_pred_x)
		#print(one_table_pred_x.shape)
		return one_table_pred_x

	def transform_to_1d_array(self, table_ids, all_pred_list, join_infos):
		join_x = self.one_hot_join_encoding(join_infos)
		table_x = self.one_hot_table_encoding(table_ids)

		pred_x = []
		for t_id, pred_list in zip(table_ids, all_pred_list):
			one_table_pred_x = self.one_table_pred_encoding(t_id, pred_list)
			pred_x.append(one_table_pred_x)
		pred_x = np.vstack(pred_x)
		return table_x, pred_x, join_x

	def one_hot_operator_encoding(self, join_infos = None):
		# For TreeLSTM operator encoding, only support table scan and equal join
		operator_x = np.zeros(shape=(2 + self.total_num_joins,))
		if not join_infos: # is a table scan operator
			operator_x[0] = 1
			return operator_x
		operator_x[1] = 1
		for join_info in join_infos:
			t1_id, t2_id, col_name, op = join_info.t1_id, join_info.t2_id, join_info.col_name, "="
			join_triple = (t1_id, t2_id, col_name) if t1_id < t2_id else (t2_id, t1_id, col_name)
			idx = self.all_join_triples.index(join_triple)
			operator_x[2 + idx] = 1
		return operator_x

	def one_hot_meta_encoding(self, table_ids):
		meta_x = np.zeros(shape=(self.num_tables,))
		np.put(meta_x, table_ids, 1)
		return meta_x


	def transform_to_1d_array_lstm(self, table_ids, all_pred_list, join_infos):

		if len(table_ids) == 1: # only one table
			meta_x = self.one_hot_meta_encoding(table_ids)
			operator_x = self.one_hot_operator_encoding()
			pred_x = self.one_table_pred_encoding(t_id=table_ids[0], pred_list=all_pred_list[0])
			#pred_x, operator_x, meta_x = self._to_torch_tensor(pred_x, operator_x, meta_x)
			#print(pred_x.shape)
			root = TreeNode(pred_x, operator_x, meta_x, level=0)
			return root
		# len(table_ids) > 0
		table_x_list, pred_x_list, join_x_list = [], [], []
		join_order, join_infos_order = self.get_join_order(table_ids, join_infos)
		all_pred_list = [all_pred_list[table_ids.index(t_id)] for t_id in join_order] # reorder the predicates
		# build the start leaf
		l = 0
		meta_x = self.one_hot_meta_encoding([join_order[l]])
		operator_x = self.one_hot_operator_encoding()
		pred_x = self.one_table_pred_encoding(t_id= join_order[l], pred_list= all_pred_list[l])
		start_leaf = TreeNode(pred_x, operator_x, meta_x, level= 0)
		root = start_leaf
		for join_infos in join_infos_order:
			l += 1
			# build a new leaf
			meta_x = self.one_hot_meta_encoding([join_order[l]])
			operator_x = self.one_hot_operator_encoding()
			pred_x = self.one_table_pred_encoding(t_id=join_order[l], pred_list=all_pred_list[l])
			leaf = TreeNode(pred_x, operator_x, meta_x, level=0)
			# build current root
			meta_x = self.one_hot_meta_encoding(join_order[0:l + 1])
			operator_x = self.one_hot_operator_encoding(join_infos)
			pred_x = np.vstack([leaf.pred_features, root.pred_features])

			new_root = TreeNode(pred_x, operator_x, meta_x, level=l)
			new_root.add_child(leaf)
			new_root.add_child(root)
			root = new_root
		return root

	def get_join_order(self, table_ids, join_infos):
		join_order = [] # a list of t_id
		join_infos_order = [] # a list of join_infos
		join_edges = []
		sample_frontier = set()
		tid_to_join_infos = dict([(t_id, []) for t_id in table_ids])
		for join_info in join_infos:
			t1_id, t2_id = join_info.t1_id, join_info.t2_id
			join_edges.append((t1_id, t2_id))
			tid_to_join_infos[t1_id].append(join_info)
			tid_to_join_infos[t2_id].append(join_info)

		join_G = nx.Graph()
		join_G.add_edges_from(join_edges)
		start = random.choice(table_ids)
		join_order.append(start)
		for neighbor in join_G.adj[start]:
			sample_frontier.add(neighbor)
		while len(sample_frontier):
			cur_tid = sample_frontier.pop()
			cur_join_infos = []
			for next_tid in join_G.adj[cur_tid]:
				if next_tid in join_order:
					continue
				sample_frontier.add(next_tid)
			for join_info in tid_to_join_infos[cur_tid]:
				if join_info.t1_id in join_order or join_info.t2_id in join_order:
					cur_join_infos.append(join_info)
			join_order.append(cur_tid)
			join_infos_order.append(cur_join_infos)
		return join_order, join_infos_order


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

	def transform_to_arrays(self, all_queries, all_cards, model_type ='MSCN'):
		# model_type is equal to the args.model_type 'MSCN', 'TLSTM'
		# Transform queries to numpy array
		num_queries = len(all_queries)
		X = []
		if model_type == 'MSCN':
			for (table_ids, all_pred_list, join_infos) in all_queries:
				table_x, pred_x, join_x = self.transform_to_1d_array(table_ids, all_pred_list, join_infos)
				X.append((table_x, pred_x, join_x))
		else: # model_type == 'TLSTM'
			for (table_ids, all_pred_list, join_infos) in all_queries:
				root = self.transform_to_1d_array_lstm(table_ids, all_pred_list, join_infos)
				X.append(root)
		Y = np.reshape(np.array(all_cards), newshape=(num_queries, 1))
		Y = np.log2(Y)
		return X, Y


class TreeNode(object):
	def __init__(self, pred_features, op_features, meta_features, level):
		self.pred_features = pred_features # [num_pred, num_pred_feat]
		self.op_features = op_features # [2 + self.total_num_joins, ]
		self.meta_features = meta_features # [num_table, ]
		self.level = level
		self.children = []

	def add_child(self, child):
		self.children.append(child)

	def recursive_to_torch_tensor(self, cuda):
		for child in self.children:
			child.recursive_to_torch_tensor(cuda)
		if cuda:
			self.pred_features = torch.FloatTensor(self.pred_features).cuda().unsqueeze(dim=0)
			self.op_features = torch.FloatTensor(self.op_features).cuda().unsqueeze(dim=0)
			self.meta_features = torch.FloatTensor(self.meta_features).cuda().unsqueeze(dim=0)
		else:
			self.pred_features = torch.FloatTensor(self.pred_features).unsqueeze(dim=0)
			self.op_features = torch.FloatTensor(self.op_features).unsqueeze(dim=0)
			self.meta_features = torch.FloatTensor(self.meta_features).unsqueeze(dim=0)