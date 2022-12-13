import pandas as pd
import os
from baselines.encoder import MSCNEncoder, MSCNJoinQueryEncoder
from JoinQuerySampler import BinaryJoinQuerySampler, Table
from QuerySampler import GeneralQuerySampler

### TPC-H Tables
TPCH_CLEAN_DATA_DIR = "/home/kfzhao/data/rdb/TPCH_clean/"

def LoadTPCH_lineitem(data_path, filename="lineitem.csv", nrows = None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['order_key', 'part_key', 'supp_key', 'line_number', 'quantity', 'extended_price', 'discount', 'tax']
	col_types = ['numerical'] * 8
	#df_dataset = pd.read_csv(csv_file, header=None, delimiter='|', usecols=[0, 1, 2, 3, 4, 5, 6, 7], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = ''
	return df_dataset, col_types, primary_key


def LoadTPCH_part(data_path, filename="part.csv", nrows = None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['part_key', 'size', 'retail_price']
	col_types = ['numerical'] * 3
	#df_dataset = pd.read_csv(csv_file, header=None, delimiter='|', usecols=[0, 5, 7], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'part_key'
	return df_dataset, col_types, primary_key

def LoadTPCH_orders(data_path, filename="orders.csv", nrows = None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['order_key', 'order_status', 'total_price', 'ship_priority']
	#col_types = ['numerical', 'categorical', 'numerical', 'categorical']
	col_types = ['numerical'] * 4
	#df_dataset = pd.read_csv(csv_file, header=None, delimiter='|', usecols=[0, 2, 3, 7], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'order_key'
	return df_dataset, col_types, primary_key

def LoadTPCH_supplier(data_path, filename="supplier.csv", nrows = None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['supp_key', 'nationkey', 'acctbal']
	#col_types = ['numerical', 'categorical', 'numerical']
	col_types = ['numerical'] * 3
	#df_dataset = pd.read_csv(csv_file, header=None, delimiter='|', usecols=[0, 3, 5], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'supp_key'
	return df_dataset, col_types, primary_key

### TPC-DS Tables
TPCDS_CLEAN_DATA_DIR = "/home/kfzhao/data/rdb/TPCDS_clean"

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


### Yelp Tables
YELP_CLEAN_DATA_DIR = "/home/kfzhao/data/rdb/yelp_clean"

def LoadYelp_Business_raw(data_path, filename="business.csv", nrows=None):
	# [209,394 rows x 8 columns]
	csv_file = os.path.join(data_path, filename)
	#col_names = ['business_id', 'city', 'state', 'latitude', 'longitude', 'stars', 'review_count']
	#col_types = ['categorical', 'categorical', 'categorical', 'numerical', 'numerical', 'numerical', 'numerical']
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=';',usecols=[0, 3, 4, 6, 7, 8, 9], names=col_names, nrows=nrows)

	col_names = ['business_id', 'latitude', 'longitude', 'business_stars', 'business_review_count']
	#col_types = ['categorical', 'numerical', 'numerical', 'numerical', 'numerical']
	col_types = ['numerical'] * 5
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=';',usecols=[0, 6, 7, 8, 9], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'business_id'
	return df_dataset, col_types, primary_key


def LoadYelp_Reviews_raw(data_path, filename="review.csv", nrows=None):
	# [8,021,122 rows x 7 columns]
	csv_file = os.path.join(data_path, filename)
	col_names = ['review_id', 'user_id', 'business_id', 'review_stars', 'review_useful', 'review_funny', 'review_cool']
	#col_types = ['categorical', 'categorical', 'categorical', 'numerical', 'numerical', 'numerical', 'numerical']
	col_types = ['numerical'] * 7
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=';',usecols=[0, 1, 2, 3, 4, 5, 6], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names, nrows=nrows)
	primary_key = 'review_id'
	return df_dataset, col_types, primary_key



def LoadYelp_Users_raw(data_path, filename="user.csv", nrows=None):
	# [1,968,703 rows x 18 columns]
	csv_file = os.path.join(data_path, filename)
	col_names = [
	'user_id',
	'user_review_count',
	'user_useful',
	'user_funny',
	'user_cool',
	'fans',
	'average_stars',
	'compliment_hot',
	'compliment_more',
	'compliment_profile',
	'compliment_cute',
	'compliment_list',
	'compliment_note',
	'compliment_plain',
	'compliment_cool',
	'compliment_funny',
	'compliment_writer',
	'compliment_photos'
	]
	#col_types = ['categorical'] + ['numerical'] * 17
	col_types = ['numerical'] * 18
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', usecols=[0, 2, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], names=col_names,nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';',names=col_names, nrows=nrows)
	primary_key = 'user_id'
	return df_dataset, col_types, primary_key



def LoadYelp_Reviews(data_path, filename="review_tmp.csv", nrows=None):
	# [8,021,122 rows x 7 columns]
	csv_file = os.path.join(data_path, filename)
	col_names = ['review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny', 'cool']
	col_types = ['categorical', 'categorical', 'categorical', 'numerical', 'numerical', 'numerical', 'numerical']
	#df_dataset = pd.read_csv(csv_file, header=0, delimiter=';',usecols=[0, 1, 2, 3, 4, 5, 6], names=col_names, nrows=nrows)
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';', names=col_names,
							 nrows=nrows)
	# df_dataset.convert_dtypes()
	return df_dataset, col_types

def LoadYelp_Users(data_path, filename="user_tmp.csv", nrows=None):
	# [1,968,703 rows x 18 columns]
	csv_file = os.path.join(data_path, filename)
	col_names = [
	'user_id',
#	'name',
	'review_count',
#	'yelping_since',
	'useful',
	'funny',
	'cool',
#	'elite',
#	'friends',
	'fans',
	'average_stars',
	'compliment_hot',
	'compliment_more',
	'compliment_profile',
	'compliment_cute',
	'compliment_list',
	'compliment_note',
	'compliment_plain',
	'compliment_cool',
	'compliment_funny',
	'compliment_writer',
	'compliment_photos'
	]
	col_types = ['categorical'] + ['numerical'] * 17
	df_dataset = pd.read_csv(csv_file, header=0, delimiter=';',
							 #usecols=[0, 2, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
							 names=col_names,
							 nrows=nrows)
	return df_dataset, col_types


def LoadSales(data_path, filename="train.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['store', 'item', 'sales', 'promote']
	col_types = ['categorical', 'categorical', 'numerical', 'categorical']
	df_dataset = pd.read_csv(csv_file, header=0, usecols=[2, 3, 4, 5], names=col_names, nrows=nrows)
	#df_dataset.convert_dtypes()
	return df_dataset, col_types


def LoadHiggs(data_path, filename="HIGGS.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
	col_types = ['numerical'] * len(col_names)
	df_dataset = pd.read_csv(csv_file, header=None, usecols=[22, 23, 24, 25, 26, 27, 28],
							 names=col_names, nrows=nrows)
	return df_dataset, col_types

def LoadForest(data_path, filename="forest.csv", nrows=None):
	csv_file = os.path.join(data_path, filename)
	col_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
	col_types = ['numerical'] * len(col_names)
	df_dataset = pd.read_csv(csv_file, header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
							 names=col_names, nrows=nrows)
	return df_dataset, col_types


def load_training_data(args):
	chunk_size = args.chunk_size
	data_path =args.data_path
	relations = args.relations.split(',')
	relations = [relation.strip() for relation in relations]
	names = args.names.split(',')
	names = [name.strip() for name in names]
	query_path = args.query_path
	encode = args.feat_encode
	nrows = 100000 if args.join_query else None
	tables = []

	for relation, name in zip(relations, names):
		if relation == 'forest':
			df_dataset, col_types = LoadForest(data_path, nrows=nrows)
		elif relation == 'higgs':
			df_dataset, col_types = LoadHiggs(data_path, nrows=nrows)
		elif relation == 'sales':
			df_dataset, col_types = LoadSales(data_path, nrows=nrows)
		elif relation == 'yelp-review':
			df_dataset, col_types = LoadYelp_Reviews(data_path, nrows=nrows)
		elif relation == 'yelp-user':
			df_dataset, col_types = LoadYelp_Users(data_path, nrows=nrows)
		else:
			assert False, "Unsupported Dataset"
		tables.append((df_dataset, col_types, name))

	if len(tables) == 1:
		df_dataset, col_types, name = tables[0]
		if encode == 'dnn-encoder':
			query_loader = GeneralQuerySampler(df_dataset, col_types, name, chunk_size)
		else : # one-hot
			table = Table(df_dataset, col_types, name, chunk_size)
			query_loader = MSCNEncoder(table)
	else:
		df_dataset1, col_types1, name1 = tables[0]
		df_dataset2, col_types2, name2 = tables[1]
		table1 = Table(df_dataset1, col_types1, name1, chunk_size)
		table2 = Table(df_dataset2, col_types2, name2, chunk_size)
		if encode == 'dnn-encoder':
			query_loader = BinaryJoinQuerySampler(table1, table2)
		else : # one-hot
			query_loader = MSCNJoinQueryEncoder(table1, table2)
	all_queries, all_cards, all_query_infos = query_loader.load_queries(query_path)
	X, Y = query_loader.transform_to_arrays(all_queries, all_cards)
	return X, Y, all_query_infos





