import os
import datasets
import pandas as pd
from JoinQuerySampler import Table, MultiJoinQuerySampler
from baselines.encoder import MultiJoinQueryEncoder


def schema_cleaning(df_dataset_list, col_types_list, table_name_list, pk_name_list):
	pk_code_lists = []
	# map primary key column to categorical encoding
	for df, col_types, primary_key in zip(df_dataset_list, col_types_list, pk_name_list):
		if not primary_key: # empty string denotes no primary key
			pk_code_lists.append({})
			continue
		print(df.columns)
		cate = pd.Categorical(df[primary_key])
		code_dict = dict([(category, code) for code, category in enumerate(cate.categories)])
		pk_code_lists.append(code_dict)
		df[primary_key] = cate.codes
	# replace foreign key column to same categorical encoding of their primary key
	for t1_id, primary_key in enumerate(pk_name_list):
		for t2_id, df in enumerate(df_dataset_list):
			if t1_id == t2_id:
				continue
			print(t1_id, t2_id, pk_name_list[t1_id])
			if primary_key in df.columns:
				print("key value replace", t1_id, t2_id, pk_name_list[t1_id])
				df[primary_key] = df[primary_key].map(pk_code_lists[t1_id])  # map is much faster than replace
		# df[primary_key].replace(self.pk_code_lists[t1_id], inplace =True)
	# replace non-key categorical column to code
	for df, col_types in zip(df_dataset_list, col_types_list):
		for col_idx, col_name in enumerate(df.columns):
			if col_types[col_idx] == 'categorical' and col_name not in pk_name_list:
				df[col_name] = pd.Categorical(df[col_name]).codes
	schema_save_path = "/home/kfzhao/data/rdb/imdb_clean2"
	for df, table_name in zip(df_dataset_list, table_name_list):
		df.fillna(-1, inplace=True)
		df = df.astype(int) # for IMDB dataset
		df.to_csv(path_or_buf=os.path.join(schema_save_path, "{}.csv".format(table_name)), sep=';', index=False)



class DBSchema(object):
	def __init__(self, df_dataset_list, col_types_list, table_name_list, primary_key_list, chunk_size):
		self.primary_key_list = primary_key_list
		self.pk_code_lists = []
		# map primary key column to categorical encoding
		for df, col_types, primary_key in zip(df_dataset_list, col_types_list, primary_key_list):
			if not primary_key:
				self.pk_code_lists.append({})
				continue
			cate = pd.Categorical(df[primary_key])
			code_dict = dict([(category, code) for code, category in enumerate(cate.categories)])
			self.pk_code_lists.append(code_dict)
			df[primary_key] = cate.codes

		# prepare the fk categorical code for each table
		self.fk_code_dicts_list = []
		for t2_id, df in enumerate(df_dataset_list):
			fk_code_dicts = {}
			for t1_id, key in enumerate(primary_key_list):
				if t2_id == t1_id:
					continue
				if key in df.columns:
					pk_code_dict = self.pk_code_lists[t1_id]
					fk_code_dicts[key] = pk_code_dict
			self.fk_code_dicts_list.append(fk_code_dicts)
		self.tables = []
		for df, col_types, table_name, fk_code_dicts in zip(df_dataset_list, col_types_list, table_name_list, self.fk_code_dicts_list):
			table = Table(df, col_types, table_name, fk_code_dicts=fk_code_dicts, chunk_size=chunk_size)
			self.tables.append(table)

	def print_schema_info(self):
		print("<" * 80)
		for t_id, table in enumerate(self.tables):
			print("Table {}: {}".format(t_id, table.table_name))
			print("Columns", table.df.columns)
			print("PK name: {}".format(self.primary_key_list[t_id]))
			#print("FK name: {}".format(','.join(table)))
		print(">" * 80)


def load_training_schema_data(args):
	schema_name = args.schema_name
	encode = args.feat_encode
	query_path = args.query_path
	data_path = args.data_path
	chunk_size = args.chunk_size
	df_dataset_list, col_types_list, pk_name_list = [], [], []
	local_vars = locals()
	if schema_name == 'yelp':
		load_funcs = [datasets.LoadYelp_Business_raw(data_path), datasets.LoadYelp_Reviews_raw(data_path),
					  datasets.LoadYelp_Users_raw(data_path)]
		table_name_list = ['business', 'review', 'user']
	elif schema_name ==  'tpcds':
		load_funcs = [datasets.LoadTPCDS_store_sales(data_path), datasets.LoadTPCDS_store(data_path), datasets.LoadTPCDS_item(data_path),
					  datasets.LoadTPCDS_customer(data_path), datasets.LoadTPCDS_promotion(data_path)]
		table_name_list = ['store_sales', 'store', 'item', 'customer', 'promotion']
	elif schema_name == 'tpch':
		load_funcs = [datasets.LoadTPCH_lineitem(data_path), datasets.LoadTPCH_part(data_path), datasets.LoadTPCH_orders(data_path),
					  datasets.LoadTPCH_supplier(data_path)]
		table_name_list = ['lineitem', 'part', 'orders', 'supplier']
	elif schema_name == 'imdb_simple':
		load_funcs = [datasets.LoadIMDB_title(data_path), datasets.LoadIMDB_cast_info(data_path),
					  datasets.LoadIMDB_movie_info(data_path),
					  datasets.LoadIMDB_movie_companies(data_path), datasets.LoadIMDB_movie_info_idx2(data_path),
					  datasets.LoadIMDB_movie_keyword(data_path)]
		table_name_list = ['title', 'cast_info', 'movie_info', 'movie_companies', 'movie_info_idx', 'movie_keyword']
	elif schema_name == 'imdb':
		load_funcs = [datasets.LoadIMDB_title(data_path), datasets.LoadIMDB_cast_info(data_path),
					  datasets.LoadIMDB_movie_info(data_path),
					  datasets.LoadIMDB_movie_companies(data_path), datasets.LoadIMDB_movie_info_idx(data_path),
					  datasets.LoadIMDB_movie_keyword(data_path)]
		table_name_list = ['title', 'cast_info', 'movie_info', 'movie_companies', 'movie_info_idx', 'movie_keyword']
	else:
		assert False, "Unsupported Schema!"
	for load_func in load_funcs:
		local_vars['df_dataset'], local_vars['col_types'], local_vars['pk'] = load_func
		df_dataset_list.append(local_vars['df_dataset'])
		col_types_list.append(local_vars['col_types'])
		pk_name_list.append(local_vars['pk'])

	schema = DBSchema(df_dataset_list, col_types_list, table_name_list, pk_name_list, chunk_size)
	schema.print_schema_info()
	if encode == 'dnn-encoder':
		multi_join_sampler = MultiJoinQuerySampler(schema.tables)
		all_queries, all_cards, all_query_infos = multi_join_sampler.load_queries(query_path=query_path)
		X, Y = multi_join_sampler.transform_to_arrays(all_queries, all_cards)
	elif encode == 'one-hot':
		multi_join_encoder = MultiJoinQueryEncoder(schema.tables)
		all_queries, all_cards, all_query_infos = multi_join_encoder.load_queries(query_path=query_path)
		X, Y = multi_join_encoder.transform_to_arrays(all_queries, all_cards, args.model_type)
	else: assert False, "unsupported encoder type!"
	return X, Y, all_query_infos



if __name__ == "__main__":
	#data_path = "/home/kfzhao/data/rdb/TPCDS_2Gclean"
	data_path = '/home/kfzhao/data/rdb/imdb_clean'
	df_dataset_list, col_types_list, pk_name_list = [], [], []
	#load_funcs = [datasets.LoadYelp_Business_raw(), datasets.LoadYelp_Reviews_raw(), datasets.LoadYelp_Users_raw()]
	#load_funcs = [datasets.LoadTPCDS_store_sales(data_path), datasets.LoadTPCDS_store(data_path), datasets.LoadTPCDS_item(data_path),
	#			  datasets.LoadTPCDS_customer(data_path), datasets.LoadTPCDS_promotion(data_path)]
	load_funcs = [datasets.LoadIMDB_title(data_path), datasets.LoadIMDB_cast_info(data_path), datasets.LoadIMDB_movie_info(data_path),
				  datasets.LoadIMDB_movie_companies(data_path), datasets.LoadIMDB_movie_info_idx(data_path), datasets.LoadIMDB_movie_keyword(data_path)]
	#load_funcs = [datasets.LoadTPCH_lineitem(), datasets.LoadTPCH_part(), datasets.LoadTPCH_orders(), datasets.LoadTPCH_supplier()]
	#table_name_list = ['business', 'review', 'user']
	#table_name_list = ['store_sales', 'store', 'item', 'customer', 'promotion']
	#table_name_list = ['lineitem', 'part', 'orders', 'supplier']
	table_name_list = ['title', 'cast_info', 'movie_info', 'movie_companies', 'movie_info_idx', 'movie_keyword']
	local_vars = locals()
	for load_func in load_funcs:
		local_vars['df_dataset'], local_vars['col_types'], local_vars['pk'] = load_func
		df_dataset_list.append(local_vars['df_dataset'])
		col_types_list.append(local_vars['col_types'])
		pk_name_list.append(local_vars['pk'])

	#print("schema cleaning ...")
	#schema_cleaning(df_dataset_list, col_types_list, table_name_list, pk_name_list)

	schema = DBSchema(df_dataset_list, col_types_list, table_name_list, pk_name_list, chunk_size=64)
	
	schema.print_schema_info()

	print("build multiple join sampler")
	multi_join_sampler = MultiJoinQuerySampler(schema.tables, query_engine='clickhouse')

	multi_join_sampler.sample_batch_query(save_path='./', num_tables=2, mini_batch=100, data_centric=True, cat_size=100)
	#multi_join_sampler.parallel_sampler(mini_batch=3000, data_centric=True)


	"""
	all_queries, all_cards, all_query_infos = multi_join_sampler.load_queries(
		query_path="/home/kfzhao/PycharmProjects/NNGP/queryset/join_lineitem_part_orders_supplier_10_data_centric_427")
	X, Y = multi_join_sampler.transform_to_arrays(all_queries, all_cards)
	for query_info in all_query_infos:
		print(query_info)
	print(X.shape, Y.shape)
	print(len(all_query_infos))
	"""