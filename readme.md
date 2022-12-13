Lightweight and Accurate Cardinality Estimation by Neural Network Gaussian Process
-----------------
Implementation of the Neural Network Gaussian Process (NNGP) Estimator, as described in our paper: [Lightweight and Accurate Cardinality Estimation by Neural Network Gaussian Process](https://dl.acm.org/doi/abs/10.1145/3514221.3526156)


### Project Structure
- active: Active Learning Implementations
- baselines: Several Implementations of Baselines
- dnn: Neural Network Baselines 
- Queries: An Example Query Set of Forest Dataset
- neuroestimator: A Python interface of NNGP estimator Used by PostreSQL



### Requirements
This project is tested on:
```
python 3.7
numpy
scipy
scikit-learn
pandasql
torch 1.6.0
gpytorch 1.2.1
tensorflow 2.3.1
neural-tangents 0.3
```

Set up a conda environment with depedencies installed:
```
cd path-to-repo
conda env create -f nngp.yaml
conda activate nngp
```

### Quick start with forest
Download the forest data [forest](https://archive.ics.uci.edu/ml/datasets/Covertype) and rename the csv file to 'forest.csv'.
To test forest queries:
```
python train.py
--kernel_type nngp 
--relation forest
--name forest
--query_path ./Queries/forest_data
--data_path YOUR_DATA_PATH_OF_FOREST.CSV
```

