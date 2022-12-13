## NNGP Cardinality Estimator (For PostgreSQL)

### Build and enter python environment
```
conda env create -f ntk.yaml
conda activate ntk
```

### Install estimator package
```
$neuroestimator python setup.py install
```

### Usage
An example test script is given in estimator_test.py.
```
from estimator import Estimator
### Step 1 ###
est = Estimator(schema_name, data_path, train_query_path, chunk_size=64, use_aux=True, q_error_threshold= 0, coef_var_threshold=0) 
# create the NNGP estimator, may takes several seconds to load the data and training queries
# schema_name: e.g., 'tpcds'
# data_path: absolute path of the relational csv data files 
# train_query_path: absolute path of training queries with their true cardinalities
# use_aux: whether to use the 'join_query_aux.txt'
# q_error_threshod: queries in 'join_query_aux.txt' with q_error larger than q_error_threshod well be used as training queries
# coef_var_threshod: queries in 'join_query_aux.txt' with coef_var larger than coef_var_threshold well be used as training queries
### Step 2 ###
est.load_model()
# Train the model on-the-fly, may takes several seconds
### Step 3 ###
pred_means, pred_stds = est.predict(query_lines)
# Get the predictive means and standarded deviation for a list of query strings
```

### Test Query line format
```
Line := table_name_list + ‘@’ + preds_list + ‘@’ + join_conditions
table_name_list: = table_name | [table_name + ‘,’]
preds_list := preds | [ preds + ‘@’]
# The length of table_name_list and preds_list are equal and the selection predicates are corresponds to the table with identical index
preds := numerical_pred | [numerical_pred + ‘#’]
numerical_pred := column_name + ‘,’ + upper_value + ‘,’  + lower_value
join_conditions := join_condition | [join_condition + ‘#’]
join_condition := table1_name + ’,’ +  table2_name + ’,’ + join_column_name 
	| table1_name + ’,’ +  table2_name + ’,’ + join_column_name + ‘,’ + join_operator 
	(omitted join operator means equal join)	
# currently only support PK/FK joins and join key are with the same name as 'join_column_name'
```

### File Structure
```
setup.py              # package install file
estimator_test.py     # package test script
estimator/ 
         encoder.py   # query encoder 
         estimator.py # nngp estimator
         util.py      # data/query loader, schema preprocessor, etc.         
```

