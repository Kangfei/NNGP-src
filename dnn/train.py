from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys
import os
import numpy as np
from dnn import MLP, GPRegressionModel, ExactGPModel, MultiTaskMLP, MCDropoutModel
from sklearn import neural_network
import datasets
import schemas
from torch.utils.data import DataLoader
import datetime
import torch.optim as optim
import xgboost as xgb
import gpytorch
import tqdm
from scipy.stats import  entropy
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.manifold import TSNE
import psutil
import math
import torch
from torch.utils.data.dataset import Dataset
from util import draw_uncertainty, calibration_plot, PredictionStatistics, show_memory_usage
from util import uneven_train_test_split, train_test_val_split

pred_stat = PredictionStatistics()

class QueryDataset(Dataset):
	def __init__(self, X, Y, max_classes=10):
		self.X = X
		self.Y = Y
		self.label_base = 10
		self.max_classes = max_classes

	def __len__(self):
		return self.X.shape[0]

	def __getitem__(self, index):
		x, y = self.X[index], self.Y[index]
		idx = math.ceil(math.log(math.pow(2, y), self.label_base))
		idx = self.max_classes - 1 if idx >= self.max_classes else idx
		label = torch.tensor(idx, dtype=torch.long)

		x = torch.FloatTensor(x)
		y = torch.FloatTensor(y)
		return x, y, label



def draw_embeddings(embedding, output_name, label=None):
	import seaborn as sns
	import matplotlib.pyplot as plt
	output_dir = "./{}.pdf".format(output_name)
	print("plot TSNE embedding")
	x = embedding[:, 0].reshape((embedding.shape[0],))
	y = embedding[:, 1].reshape((embedding.shape[0],))
	label = label.reshape((label.shape[0],))
	ax = sns.scatterplot(x = x, y = y, hue=label)
	plt.savefig(output_dir)

def compute_uncertainty(args, output_cal, output):
	output_cal, output = output_cal.squeeze(), output.squeeze()
	output_cal = torch.exp(output_cal) # transform output of log_softmax to probability
	if args.cuda:
		output_cal = output_cal.cpu()
		output = output.cpu()
	output = output.detach().numpy()
	output_cal = output_cal.detach().numpy()
	if args.uncertainty == "entropy":
		return entropy(output_cal, axis=-1)
	elif args.uncertainty == "confident":
		return 1.0 - np.max(output_cal, axis=-1)
	elif args.uncertainty == "margin":
		output_cal = np.sort(output_cal)
		return output_cal[:, -1] - output_cal[:, -2]
	elif args.uncertainty == "random":
		return np.random.rand(output.shape[0])
	elif args.uncertainty == "consist":
		reg_mag = np.ceil( np.log10( np.power(2.0, output)))
		cla_mag = np.argmax(output_cal, axis=-1)
		return np.power((reg_mag - cla_mag), 2)
	else:
		assert False, "Unsupported uncertainty function!"


def test_mcdropout(args, model, criterion, X_test, Y_test, query_infos_test = None):
	model.eval()
	total_loss = 0.0
	means, stds = [], []
	test_dataset = QueryDataset(X_test, Y_test, args.max_classes)
	test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
	start = datetime.datetime.now()
	for i, (X, Y, _) in enumerate(test_loader):
		if args.cuda:
			X, Y = X.cuda(), Y.cuda()
		mean, std = model.predict(X)
		loss = criterion(mean, Y)
		total_loss += loss.item()
		means.append(mean)

		stds.append(std)
	end = datetime.datetime.now()
	duration = (end - start).total_seconds()
	print('MC Dropout Test in %s seconds.' % duration)

	outputs = torch.cat(means, dim=0).unsqueeze(dim=-1)
	stds = torch.cat(stds, dim=0).unsqueeze(dim=-1)
	print("Test MSE Loss={:.4f}".format(total_loss))
	outputs = outputs.cpu().detach().numpy()
	stds = stds.cpu().detach().numpy()
	errors = outputs - Y_test
	pred_stat.get_prediction_details(errors, query_infos_test, partition_keys='num_table')
	outputs = np.ravel(outputs)
	stds = np.ravel(stds)
	stds = stds / np.max(outputs, 0)
	draw_uncertainty('tpcds_MCDropout', errors, stds, Y_test)
	all_Y_test = pred_stat.get_partitioned_data(Y_test, query_infos_test, part_keys='num_table')
	"""
	all_outputs = pred_stat.get_partitioned_data(outputs, query_infos_test, part_keys='num_table')
	all_pred_std = pred_stat.get_partitioned_data(stds, query_infos_test, part_keys='num_table')
	for (Y_test, outputs, pred_std) in zip(all_Y_test, all_outputs, all_pred_std):
		calibration_plot(Y_test, outputs, pred_std)
	"""


def train_mcdropout(args, model, optimizer, criterion, X_train, Y_train, scheduler = None):
	device = args.device
	if args.cuda:
		model.to(device)
	train_dataset = QueryDataset(X_train, Y_train, args.max_classes)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

	start = datetime.datetime.now()
	for epoch in range(args.epochs):
		total_loss = 0.0
		model.train()
		for i, (X, Y, label) in enumerate(train_loader):
			if args.cuda:
				X, Y, label = X.cuda(), Y.cuda(), label.cuda()

			optimizer.zero_grad()
			mu, sigma = model(X)
			#print(mu.shape, sigma.shape)

			#loss = model.loss(mu, Y, sigma)
			loss = criterion(mu, Y)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		print("{}-th Epochs: Train Loss={:.4f}".format(epoch, total_loss))
		if scheduler is not None and (epoch + 1) % args.decay_patience == 0:
			scheduler.step()
	end = datetime.datetime.now()
	duration = (end - start).total_seconds()
	print('MC Dropout Training in %s seconds.' % duration)
	return model


def test_mse(args, model, criterion, X_test, Y_test, query_infos_test = None):
	model.eval()
	total_loss = 0.0
	outputs = None
	outputs_cla = None
	test_dataset = QueryDataset(X_test, Y_test, args.max_classes)
	test_loader = DataLoader(test_dataset, batch_size=5000, shuffle=False)
	start = datetime.datetime.now()

	for i, (X, Y, _) in enumerate(test_loader):
		if args.cuda:
			X, Y = X.cuda(), Y.cuda()
		output, output_cal = model(X)
		loss = criterion(output, Y)
		#print(error.shape)
		total_loss += loss.item()
		outputs = output if outputs is None else torch.cat([outputs, output], dim= 0)
		outputs_cla = output_cal if outputs_cla is None else torch.cat([outputs_cla, output_cal], dim=0)
	end = datetime.datetime.now()
	duration = (end - start).total_seconds()
	print("DNN  Total Inference time={} seconds".format(duration))
	print("Test MSE Loss={:.4f}".format(total_loss))
	errors = outputs.cpu().detach().numpy() - Y_test
	pred_stat.get_prediction_details(errors, query_infos_test, partition_keys="num_table")
	"""
	uncertainty = compute_uncertainty(args, outputs_cla, outputs)
	errors = np.ravel(errors)
	outputs = np.ravel(outputs.cpu().detach().numpy())
	uncertainty = np.ravel(uncertainty)
	draw_uncertainty("dnn-{}".format(args.uncertainty), errors, uncertainty, outputs)
	"""



def test_accuracy(args, model, criterion, test_dataloader):
	model.eval()
	batch_size = test_dataloader.batch_size
	correct, total = 0.0, 0.0

	for i, (X, Y) in enumerate(test_dataloader):
		if args.cuda:
			X, Y = X.cuda(), Y.cuda()
		pred = model(X)
		pred = pred.argmax(dim=1)
		Y = Y.argmax(dim = 1)
		total += X.shape[0]
		correct += (pred == Y).sum().item()
	print("# Correct:{}, # total: {}".format(correct, total))
	print("Test accuracy = {}".format(correct/total))




def train(args, model, optimizer, criterion, criterion_cal,
		  X_train, Y_train, scheduler = None):
	device = args.device
	if args.cuda:
		model.to(device)
	train_dataset = QueryDataset(X_train, Y_train, args.max_classes)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

	max_memory = 0
	start = datetime.datetime.now()
	for epoch in range(args.epochs):
		total_loss = 0.0
		model.train()
		for i, (X, Y, label) in enumerate(train_loader):
			if args.cuda:
				X, Y, label = X.cuda(), Y.cuda(), label.cuda()

			optimizer.zero_grad()
			output, output_cla = model(X)
			#print(output.shape)

			loss = criterion(output, Y) + args.coeff * criterion_cal(output_cla, label)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
			max_memory = max(max_memory, float(psutil.virtual_memory().used/(1024**3)))
		print("{}-th Epochs: Train MSE Loss={:.4f}".format(epoch, total_loss))
		if scheduler is not None and (epoch + 1) % args.decay_patience == 0:
			scheduler.step()
	end = datetime.datetime.now()
	duration = (end - start).total_seconds()
	print('DNN Training in %s seconds.' % duration)
	print('memory usage:', str(max_memory)[:5])
	return model



def main(args):
	#X, Y, all_query_infos = datasets.load_training_data(args)
	X, Y, all_query_infos = schemas.load_training_schema_data(args)
	num_queries, num_feats = X.shape

	print("number of query: {}".format(num_queries))

	X_train, Y_train, query_infos_train, X_test, Y_test, query_infos_test, X_val, Y_val, query_infos_val = \
		train_test_val_split(X, Y, train_frac=0.6, test_frac=0.2, all_query_infos=all_query_infos)

	#X_train, Y_train, query_infos_train, X_test, Y_test, query_infos_test, X_val, Y_val, query_infos_val = \
	#	uneven_train_test_split(X, Y, all_query_infos=all_query_infos, skew_split_keys='num_predicates', train_frac=0.8,	skew_ratio=0.2)

	print(X_train.shape, X_test.shape)
	print(Y_train.shape, Y_test.shape)

	criterion = torch.nn.MSELoss()
	criterion_cla = torch.nn.NLLLoss()

	if args.model_type == 'DKL':
		DKL_train_and_test(args, X_train, Y_train, X_test, Y_test)
	elif args.model_type == 'GP':
		#ExactGP_train_and_test(args, X_train, Y_train, X_test, Y_test, query_infos_test)
		sklearnGP_train_and_test(X_train, Y_train, X_test, Y_test, query_infos_test)
	elif args.model_type == 'KRR':
		KRR_train_and_test(X_train, Y_train, X_test, Y_test)
	elif args.model_type == 'DNN':
		# define the model
		model = MultiTaskMLP(in_ch=num_feats, hid_ch=args.num_hid, reg_out_ch=1, cla_out_ch=args.max_classes)
		optimizer = optim.Adam(model.parameters(),
							   lr=args.learning_rate, weight_decay=args.weight_decay)
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_factor)
		model = train(args, model, optimizer, criterion, criterion_cla, X_train, Y_train, scheduler)
		test_mse(args, model, criterion, X_test, Y_test, query_infos_test)
	elif args.model_type == 'XGB':
		#xgb_train_and_test(X_train, Y_train, X_test, Y_test, query_infos_test)
		xgb_train_and_test(X_train, Y_train, X_train, Y_train, query_infos_train)
	elif args.model_type == 'MLP':
		mlp_train_and_test(X_train, Y_train, X_test, Y_test, args.num_hid, args.epochs, args.batch_size, args.learning_rate, args.weight_decay)
	elif args.model_type == 'MCDropout':
		model = MCDropoutModel(input_dim=num_feats, output_dim=1, hid_dim=args.num_hid)
		optimizer = optim.Adam(model.parameters(),
							   lr=args.learning_rate, weight_decay=args.weight_decay)
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_factor)
		model = train_mcdropout(args, model, optimizer, criterion, X_train, Y_train, scheduler)
		test_mcdropout(args, model, criterion, X_test, Y_test, query_infos_test)
	else:
		assert False, "Unsupported model type!"



def mlp_train_and_test(X_train, Y_train, X_test, Y_test, num_hid, epochs, batch_size, lr, weight_decay):
	# Instantiation
	mlp_reg = neural_network.MLPRegressor(hidden_layer_sizes=num_hid, activation='relu',
										  solver='adam', alpha=weight_decay, batch_size=batch_size,
										  learning_rate='constant', learning_rate_init=lr,
										  power_t=0.5, max_iter=epochs, shuffle=True)
	# Fit the model
	Y_train, Y_test = Y_train.ravel(), Y_test.ravel()
	start = datetime.datetime.now()
	mlp_reg.fit(X_train, Y_train)
	end = datetime.datetime.now()
	duration = (end - start).total_seconds()
	print('MLP Training in %s seconds.' % duration)

	# Predict the model
	pred = mlp_reg.predict(X_test)
	mse = mean_squared_error(Y_test, pred)
	print("MLP mean square error: {:.4f}".format(mse))

	errors = pred - Y_test
	pred_stat.get_prediction_details(errors)



def xgb_train_and_test(X_train, Y_train, X_test, Y_test, query_infos_test):
	# Instantiation
	xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', tree_method="hist", grow_policy="lossguide",
							n_estimators=32, seed=123)
	# Fit the model
	start = datetime.datetime.now()
	xgb_reg.fit(X_train, Y_train)
	show_memory_usage(cuda=False)
	end = datetime.datetime.now()
	duration = (end - start).total_seconds()
	print('XGBoost Training in %s seconds.' % duration)

	# Predict the model
	start = datetime.datetime.now()
	pred = xgb_reg.predict(X_test)
	show_memory_usage(cuda=False)
	end = datetime.datetime.now()
	duration = (end - start).total_seconds()
	print('XGBoost Prediction in %s seconds.' % duration)
	mse = mean_squared_error(Y_test, pred)
	print("xgb mean square error: {:.4f}".format(mse))
	errors = pred - Y_test
	pred_stat.get_prediction_details(errors, query_infos_test, partition_keys='num_predicates')


def DKL_train_and_test(args, X_train, Y_train, X_test, Y_test):
	X_train, Y_train = torch.FloatTensor(X_train), torch.FloatTensor(Y_train).reshape((Y_train.shape[0]))
	X_test, Y_test = torch.FloatTensor(X_test), torch.FloatTensor(Y_test).reshape((Y_test.shape[0]))

	feature_extractor = MLP(in_ch=X_train.shape[1], hid_ch=args.num_hid, out_ch=2)
	likelihood = gpytorch.likelihoods.GaussianLikelihood()
	model = GPRegressionModel(X_train, Y_train, likelihood, feature_extractor)
	if args.cuda :
		model = model.cuda()
		likelihood = likelihood.cuda()
		X_train, Y_train, X_test, Y_test = X_train.cuda(), Y_train.cuda(), X_test.cuda(), Y_test.cuda()

	model.train()
	likelihood.train()
	optimizer = torch.optim.Adam([
		{'params': model.feature_extractor.parameters()},
		{'params': model.covar_module.parameters()},
		{'params': model.mean_module.parameters()},
		{'params': model.likelihood.parameters()},
	], lr=0.01)
	mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
	iterator = tqdm.tqdm(range(args.epochs))
	for i in iterator:
		optimizer.zero_grad()
		output = model(X_train)
		loss = -mll(output, Y_train)
		#print(output, Y_train, loss)
		loss.backward()
		print("{}-th Epochs: DKL Train Loss={:.4f}".format(i, loss.item()))
		optimizer.step()

	model.eval()
	likelihood.eval()
	preds = model(X_test)
	print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - Y_test))))
	errors = (preds.mean - Y_test).cpu().detach().numpy()
	pred = preds.mean.cpu().detach().numpy()
	Y_test = Y_test.cpu().detach().numpy()
	for i in range(pred.shape[0]):
		print("True card {}, Estimated card {}, errors {}.".format(Y_test[i], pred[i], errors[i]))
	pred_stat.get_prediction_details(errors)

def ExactGP_train_and_test(args, X_train, Y_train, X_test, Y_test, query_infos_test):
	X_train, Y_train = torch.FloatTensor(X_train), torch.FloatTensor(Y_train).reshape((Y_train.shape[0]))
	X_test, Y_test = torch.FloatTensor(X_test), torch.FloatTensor(Y_test).reshape((Y_test.shape[0]))

	# initialize likelihood and model
	likelihood = gpytorch.likelihoods.GaussianLikelihood()
	model = ExactGPModel(X_train, Y_train, likelihood)

	if args.cuda :
		model = model.cuda()
		likelihood = likelihood.cuda()
		X_train, Y_train, X_test, Y_test = X_train.cuda(), Y_train.cuda(), X_test.cuda(), Y_test.cuda()

	model.train()
	likelihood.train()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
	mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
	iterator = tqdm.tqdm(range(args.epochs))
	for i in iterator:
		optimizer.zero_grad()
		output = model(X_train)
		loss = -mll(output, Y_train)
		#print(output, Y_train, loss)
		loss.backward()
		print("{}-th Epochs: GP Train Loss={:.4f}".format(i, loss.item()))
		optimizer.step()

	model.eval()
	likelihood.eval()
	start = datetime.datetime.now()
	preds = model(X_test)
	end = datetime.datetime.now()
	duration = (end - start).total_seconds()
	print("Exact GP Total Inference time={} seconds".format(duration))
	print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - Y_test))))
	errors = (preds.mean - Y_test).cpu().detach().numpy()
	pred_stat.get_prediction_details(errors, query_infos_test, partition_keys='num_predicates')

def sklearnGP_train_and_test(X_train, Y_train, X_test, Y_test, query_infos_test):
	from sklearn.gaussian_process import GaussianProcessRegressor
	from sklearn.gaussian_process.kernels import RBF
	Y_train, Y_test = Y_train.ravel(), Y_test.ravel()
	kernel = RBF()
	start = datetime.datetime.now()
	gp_reg = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X_train, Y_train)
	end = datetime.datetime.now()
	duration = (end - start).total_seconds()
	print("Exact GP Training time={} seconds".format(duration))

	start = datetime.datetime.now()
	pred_mean, pred_std = gp_reg.predict(X_test, return_std=True)
	end = datetime.datetime.now()
	duration = (end - start).total_seconds()
	print("Exact GP Total Inference time={} seconds".format(duration))
	errors = (pred_mean - Y_test)
	pred_stat.get_prediction_details(errors, query_infos_test, partition_keys='num_table')

def KRR_train_and_test(X_train, Y_train, X_test, Y_test):
	from sklearn.kernel_ridge import KernelRidge
	Y_train = Y_train.reshape((Y_train.shape[0]))
	Y_test = Y_test.reshape((Y_test.shape[0]))
	clf = KernelRidge(alpha=1.0)
	clf.fit(X_train, Y_train)
	preds =clf.predict(X_test)
	errors = (preds - Y_test)
	pred_stat.get_prediction_details(errors)


if __name__ == "__main__":
	parser = ArgumentParser("DNN estimator", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
	parser.add_argument("--chunk_size", default=64, type=int, help="dimension of factorized encoding")
	parser.add_argument("--num_hid", default=512, type=int, help="number of hidden")
	parser.add_argument("--model_type", type=str, default='DNN', help="DNN, DKL, GP, XGB, KRR, MCDropout")
	parser.add_argument("--feat_encode", type=str, default='dnn-encoder', help='dnn-encoder,one-hot')

	# Training parameters
	parser.add_argument("--epochs", default=80, type=int)
	parser.add_argument("--learning_rate", default=1e-4, type=float)
	parser.add_argument("--batch_size", default=32, type=int, help="batch size")
	parser.add_argument('--weight_decay', type=float, default=2e-4,
						help='Weight decay (L2 loss on parameters).')
	parser.add_argument('--decay_factor', type=float, default=0.85,
						help='decay rate of (gamma).')
	parser.add_argument('--decay_patience', type=int, default=10,
						help='num of epoches for one lr decay.')
	parser.add_argument('--no-cuda', action='store_true', default=True,
						help='Disables CUDA training.')

	parser.add_argument("--multi_task", default=True, type=bool,
						help="enable/disable card classification task.")
	parser.add_argument("--max_classes", default=10, type=int,
						help="number classes for the card classification task.")
	parser.add_argument('--coeff', type=float, default=0.5,
						help='coefficient for the classification loss.')
	parser.add_argument("--uncertainty", default="consist", type=str,
						help="The uncertainty type")  # entropy, margin, confident, consist, random are tested

	# input dir
	parser.add_argument("--relations", type=str, default='forest')
	parser.add_argument("--names", type=str, default='forest')
	parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/forest_data')

	#parser.add_argument("--relations", type=str, default='higgs')
	#parser.add_argument("--names", type=str, default='higgs')
	#parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/higgs')

	#parser.add_argument("--relations", type=str, default='yelp-review,yelp-user') #'yelp-user'
	#parser.add_argument("--names", type=str, default='review,user')
	#parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/join_review_user_100_2')

	#parser.add_argument("--relations", type=str, default='forest,forest')  # 'forest,forest'
	#parser.add_argument("--names", type=str, default='forest1,forest2')
	#parser.add_argument("--query_path", type=str,
	#					default='/home/kfzhao/PycharmProjects/NNGP/queryset/join_forest1_forest2')

	parser.add_argument("--schema_name", type=str, default='imdb_simple', help='yelp, tpcds, tpch')

	#parser.add_argument("--query_path", type=str, default="/home/kfzhao/PycharmProjects/NNGP/queryset/join_business_review_user_10_data_centric")
	#parser.add_argument("--query_path", type=str,  default="/home/kfzhao/PycharmProjects/NNGP/queryset/join_store_sales_store_item_customer_promotion_10_data_centric_427")
	#parser.add_argument("--query_path", type=str,  default="/home/kfzhao/PycharmProjects/NNGP/queryset/join_lineitem_part_orders_supplier_10_data_centric_427")
	parser.add_argument("--query_path", type=str,
						default='/home/kfzhao/PycharmProjects/NNGP/queryset/join_title_cast_info_movie_info_movie_companies_movie_info_idx_movie_keyword_10_data_centric_815')
	parser.add_argument("--data_path", type=str, default='/home/kfzhao/data/rdb/imdb_clean2')

	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	args.device = torch.device('cuda' if args.cuda else 'cpu')
	relations = args.relations.split(',')
	args.join_query = True if len(relations) > 1 else False
	print(args)
	main(args)
