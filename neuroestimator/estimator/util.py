import pandas as pd
import os
from .encoder import Table, NNGPEncoder


### TPC-DS Tables
def LoadTPCDS_store_sales(data_path, filename="store_sales.csv", nrows = None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['item_sk', 'customer_sk', 'store_sk', 'promo_sk', 'quantity', 'wholesale_cost', 'list_price', 'sales_price',
				 'ext_discount_amt', 'ext_sales_price', 'ext_wholesale_cost', 'ext_list_price', 'ext_tax', 'ext_coupon_amt',
				 'net_paid', 'net_paid_inc_tax', 'net_profit']
	col_types = ['numerical'] * 17
	#df_dataset = pd.read_csv(csv_file, header=None, delimiter='|', usecols=[2, 3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = ''
	return df_dataset, col_types, primary_key

def LoadTPCDS_store(data_path, filename = "store.csv", nrows = None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['store_sk', 'number_employees', 'floor_space', 'market_id', 'devision_id', 'company_id', 'tax_percentage']
	col_types = ['numerical'] * 7
	#df_dataset = pd.read_csv(csv_file, header=None, delimiter='|', usecols=[0, 6, 7, 10, 14, 18, 28], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'store_sk'
	return df_dataset, col_types, primary_key


def LoadTPCDS_item(data_path, filename = "item.csv", nrows = None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['item_sk', 'current_price', 'wholesale_cost', 'brand_id', 'class_id', 'category_id', 'manufact_id']
	col_types = ['numerical'] * 7
	#df_dataset = pd.read_csv(csv_file, header=None, delimiter='|', usecols=[0, 5, 6, 7, 9, 11, 13], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'item_sk'
	return df_dataset, col_types, primary_key

def LoadTPCDS_customer(data_path, filename = "customer.csv", nrows = None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['customer_sk', 'birth_day', 'birth_month', 'birth_year']
	col_types = ['numerical'] * 4
	#df_dataset = pd.read_csv(csv_file, header=None, delimiter='|', usecols=[0, 11, 12, 13], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'customer_sk'
	return df_dataset, col_types, primary_key

def LoadTPCDS_promotion(data_path, filename= "promotion.csv", nrows = None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['promo_sk', 'item_sk', 'cost', 'response_target']
	col_types = ['numerical'] * 6
	#df_dataset = pd.read_csv(csv_file, header=None, delimiter='|', usecols=[0, 4, 5, 6], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'promo_sk'
	return df_dataset, col_types, primary_key


### IMDB Tables
def LoadIMDB_title(data_path, filename="title.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['movie_id', 'kind_id', 'product_year', 'imdb_id']
	col_types = ['numerical'] * 4
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=',', usecols=[0, 3, 4, 5], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'movie_id'
	return df_dataset, col_types, primary_key

def LoadIMDB_cast_info(data_path, filename="cast_info.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['person_id', 'movie_id', 'person_role_id']
	col_types = ['numerical'] * 3
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=',', usecols=[1, 2, 3], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = ''
	return df_dataset, col_types, primary_key

def LoadIMDB_movie_info(data_path, filename="movie_info.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['movie_info_id', 'movie_id', 'info_type_id']
	col_types = ['numerical'] * 3
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=',', usecols=[0, 1, 2], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'movie_info_id'
	return df_dataset, col_types, primary_key

def LoadIMDB_movie_companies(data_path, filename="movie_companies.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['movie_id', 'company_id', 'company_type_id']
	col_types = ['numerical'] * 3
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=',', usecols=[1, 2, 3], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = ''
	return df_dataset, col_types, primary_key

def LoadIMDB_movie_info_idx(data_path, filename="movie_info_idx.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['movie_info_idx_id','movie_id', 'info_type_id']
	col_types = ['numerical'] * 3
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=',', usecols=[0, 1, 2], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'movie_info_idx_id'
	return df_dataset, col_types, primary_key

def LoadIMDB_movie_info_idx2(data_path, filename="movie_info_idx.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['movie_info_idx_id','movie_id']
	col_types = ['numerical'] * 2
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=',', usecols=[0, 1], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'movie_info_idx_id'
	return df_dataset, col_types, primary_key

def LoadIMDB_movie_keyword(data_path, filename="movie_keyword.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['movie_id', 'keyword_id']
	col_types = ['numerical'] * 2
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=',', usecols=[1, 2], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = ''
	return df_dataset, col_types, primary_key

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

def load_training_schema_data(schema_name: str, data_path: str, query_path: str, chunk_size: int, use_aux: bool, q_error_threshold: float, coef_var_threshold:float):
	assert os.path.exists(data_path), "Schema data does not exist!"
	assert os.path.exists(query_path), "Training queries do not exist!"

	df_dataset_list, col_types_list, pk_name_list = [], [], []
	local_vars = locals()
	if schema_name ==  'tpcds':
		load_funcs = [LoadTPCDS_store_sales(data_path), LoadTPCDS_store(data_path), LoadTPCDS_item(data_path),
					  LoadTPCDS_customer(data_path), LoadTPCDS_promotion(data_path)]
		table_name_list = ['store_sales', 'store', 'item', 'customer', 'promotion']
	elif schema_name == 'imdb_simple':
		load_funcs = [LoadIMDB_title(data_path), LoadIMDB_cast_info(data_path),
					  LoadIMDB_movie_info(data_path),
					  LoadIMDB_movie_companies(data_path), LoadIMDB_movie_info_idx2(data_path),
					  LoadIMDB_movie_keyword(data_path)]
		table_name_list = ['title', 'cast_info', 'movie_info', 'movie_companies', 'movie_info_idx', 'movie_keyword']
	elif schema_name == 'imdb':
		load_funcs = [LoadIMDB_title(data_path), LoadIMDB_cast_info(data_path),
					  LoadIMDB_movie_info(data_path),
					  LoadIMDB_movie_companies(data_path), LoadIMDB_movie_info_idx(data_path),
					  LoadIMDB_movie_keyword(data_path)]
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

	nngp_encoder = NNGPEncoder(schema.tables)
	all_queries, all_cards, _ = nngp_encoder.load_queries(query_path, use_aux, q_error_threshold, coef_var_threshold)
	X, Y = nngp_encoder.transform_to_arrays(all_queries, all_cards)
	return X, Y, nngp_encoder
