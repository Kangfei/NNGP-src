from estimator import Estimator
import datetime
import os
#schema_name = 'tpcds'
#data_path = '/home/kfzhao/data/rdb/TPCDS_clean'
#train_query_path = '/home/kfzhao/PycharmProjects/NNGP/queryset/join_store_sales_store_item_customer_promotion_10_data_centric_427'
#test_query_file = '/home/kfzhao/PycharmProjects/NNGP/queryset/join_store_sales_store_item_customer_promotion_10_data_centric_427/join_query_5.txt'
schema_name = 'imdb'
data_path = '/home/kfzhao/data/rdb/imdb_clean'
train_query_path = '/home/kfzhao/PycharmProjects/NNGP/queryset/join_imdb_815_FP_with_aux'
test_query_file = '/home/kfzhao/PycharmProjects/NNGP/queryset/join_title_cast_info_movie_info_movie_companies_movie_info_idx_movie_keyword_10_data_centric_815_FP/join_query_5.txt'

def load_query(test_query_file: str):
	query_lines = list()
	with open(test_query_file, 'r') as in_file:
		for line in in_file:
			terms = line.strip().split('@')
			query_line = '@'.join(terms[:-1])
			query_lines.append(query_line)
	return query_lines


# define the estimator
# load the training data and schema data, may take several seconds
est = Estimator(schema_name = schema_name, data_path = data_path, train_query_path = train_query_path,
				chunk_size=64, use_aux=False, q_error_threshold= 100.0, coef_var_threshold=1.0)

# train the model on-the-fly, may take several seconds
est.load_model()

#q1 = 'store_sales,store,item,customer,promotion@net_paid_inc_tax,8043.0,2435.59@market_id,8,8@current_price,3.4,1.98@birth_year,1932.0,-1.0@cost,1000.0,1000.0@store_sales,store,store_sk#store_sales,customer,customer_sk#store_sales,item,item_sk#store_sales,promotion,promo_sk'
#q2 = 'store_sales,store,item,promotion@ext_list_price,11147.31,2595.94#ext_coupon_amt,0.0,0.0#net_paid,3117.12,482.4@@category_id,8.0,1.0@response_target,1.0,1.0@store_sales,item,item_sk#item,promotion,item_sk#store_sales,store,store_sk'

#query_lines = [q1, q2] * 1000
query_lines = load_query(test_query_file)
start = datetime.datetime.now()
pred_mean, pred_std = est.predict(query_lines)
end = datetime.datetime.now()
duration = (end - start).total_seconds()
print("Inference time={} seconds".format(duration))


print(pred_mean.shape)
print(pred_std.shape)
