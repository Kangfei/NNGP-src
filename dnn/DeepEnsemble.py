import torch
import sys
sys.path.append('/home/kfzhao/PycharmProjects/NNGP')
import numpy as np
import random
import datetime
import datasets
import schemas
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from torch.utils.data import DataLoader
from dnn.train import QueryDataset
from dnn import MLPDensityRegressor
import torch.optim as optim
from util import draw_uncertainty, calibration_plot, train_test_val_split, PredictionStatistics

pred_stat = PredictionStatistics()

class DeepEnsemble(object):
	def __init__(self, args):
		self.args = args
		self.budget = args.budget
		self.batch_size = args.batch_size
		self.active_iters = args.active_iters
		self.active_epochs = args.active_epochs
		self.biased_sample = args.biased_sample
		self.ensemble_num = args.ensemble_num

	def train(self, models, optimizers, X_train, Y_train, schedulers, active = False):
		epochs = self.active_epochs if active else self.args.epochs
		train_dataset = QueryDataset(X_train, Y_train)
		train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
		model_cnt = 1
		#criterion = torch.nn.GaussianNLLLoss() only support in pytorch 1.9
		start = datetime.datetime.now()
		for model, optimizer, scheduler in zip(models, optimizers, schedulers):
			print("Training the {}-th model in DeepEnsemble.".format(model_cnt))
			if self.args.cuda:
				model.to(self.args.device)
			for epoch in range(epochs):
				total_loss = 0.0
				model.train()
				for i, (X, Y, _) in enumerate(train_loader):
					if self.args.cuda:
						X, Y = X.cuda(), Y.cuda()
					optimizer.zero_grad()
					mu, sigma_pos = model(X)
					#print(mu.shape, sigma_pos.shape, Y.shape)
					#print(torch.isnan(mu), torch.isnan(sigma_pos))
					#print(mu, sigma_pos)
					loss = model.loss(Y, mu, sigma_pos)
					#loss = criterion(mu, Y, sigma_pos)
					#print(loss.item())
					loss.backward()
					optimizer.step()
					total_loss += loss.item()
				print("{}-th Epochs: Train NLL Loss={:.4f}".format(epoch, total_loss))
				if scheduler is not None and (epoch + 1) % self.args.decay_patience == 0:
					scheduler.step()
			model_cnt += 1
		end = datetime.datetime.now()
		elapse_time = (end - start).total_seconds()
		print("Training time: {:.4f}s".format(elapse_time))
		return models, optimizers, schedulers

	def test(self, models, X_test, Y_test, query_infos_test = None):
		all_mu = np.zeros(shape=(X_test.shape[0], self.ensemble_num))
		all_sig_pos = np.zeros(shape=(X_test.shape[0], self.ensemble_num))
		eval_dataset = QueryDataset(X_test, Y_test)
		eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
		#criterion = torch.nn.GaussianNLLLoss()
		for model_cnt, model in enumerate(models):
			print("Evaluating the {}-th model in DeepEnsemble.".format(model_cnt))
			total_loss = 0.0
			mu_output = None
			sig_pos_output = None
			if self.args.cuda:
				model.to(self.args.device)
			for i, (X, Y, _) in enumerate(eval_loader):
				if self.args.cuda:
					X, Y = X.cuda(), Y.cuda()
				mu, sigma_pos = model(X)
				loss = model.loss(Y, mu, sigma_pos)
				#loss = criterion(mu, Y, sigma_pos)
				total_loss += loss.item()
				mu_output = mu if mu_output is None else torch.cat([mu_output, mu], dim=0)
				sig_pos_output = sigma_pos if sig_pos_output is None else torch.cat([sig_pos_output, sigma_pos], dim=0)
			print("Evaluation NLL Loss = {:.4f}".format(total_loss))
			mu_output = mu_output.cpu().detach().numpy().ravel()
			sig_pos_output = sig_pos_output.cpu().detach().numpy().ravel()
			all_mu[:, model_cnt] = mu_output
			all_sig_pos[:, model_cnt] = sig_pos_output
		final_mu = np.mean(all_mu, axis= 1)
		final_sigma = np.sqrt(np.mean(all_sig_pos + np.square(all_mu), axis=1) - np.square(final_mu))
		Y_test = Y_test.ravel()
		errors = final_mu - Y_test
		mse = np.mean(np.power(errors, 2.0))
		print("Test MSE Loss:{}".format(mse))
		if query_infos_test is not None:
			pred_stat.get_prediction_details(errors, query_infos_test, partition_keys='num_predicates')
		### draw uncertainty
		#print(errors.shape)
		"""
		errors = np.ravel(errors)
		outputs = np.ravel(final_mu)
		pred_std = np.ravel(final_sigma)
		pred_std = pred_std / np.max(outputs, 0)
		#print(errors.shape, outputs.shape, pred_std.shape)
		#draw_uncertainty('{}_deep-ensemble'.format(args.schema_name), errors, pred_std, Y_test)
		draw_uncertainty('tpcds_deep-ensemble', errors, pred_std, Y_test)
		all_Y_test = pred_stat.get_partitioned_data(Y_test, query_infos_test, part_keys='num_table')
		all_outputs = pred_stat.get_partitioned_data(outputs, query_infos_test, part_keys='num_table')
		all_pred_std = pred_stat.get_partitioned_data(pred_std, query_infos_test, part_keys='num_table')
		for (Y_test, outputs, pred_std) in zip(all_Y_test, all_outputs, all_pred_std):
			calibration_plot(Y_test, outputs, pred_std)
		#calibration_plot(Y_test, outputs, pred_std)
		"""
		return final_mu, final_sigma

	def active_test(self, models, X_test, Y_test):
		num_test = X_test.shape[0]
		final_mu, final_sigma = self.test(models, X_test, Y_test)
		final_sigma = final_sigma.ravel()
		sigma_prob = final_sigma / np.sum(final_sigma)
		num_selected = self.budget if num_test > self.budget else num_test
		indices = np.random.choice(a=num_test, size=num_selected, replace=False, p=sigma_prob) \
			if self.biased_sample else np.argsort(final_sigma)[- num_selected:]
		return indices

	def merge_data(self, select_indices, X_train, Y_train, X_test, Y_test):
		X_delta, Y_delta = X_test[select_indices], Y_test[select_indices]
		X_train_new = np.vstack((X_train, X_delta))
		Y_train_new = np.vstack((Y_train, Y_delta))
		num_test = X_test.shape[0]
		indices = np.array(list(range(num_test)))
		keep_indices = np.setdiff1d(indices, np.asarray(select_indices))
		X_test_new, Y_test_new = X_test[keep_indices], Y_test[keep_indices]
		return X_train_new, Y_train_new, X_test_new, Y_test_new

	def active_train(self, models, optimizers, schedulers,
					 X_train, Y_train, X_test, Y_test, X_val = None, Y_val= None, query_infos_val = None, pretrain = True):
		if pretrain:
			models, optimizers, schedulers = self.train(models, optimizers, X_train, Y_train, schedulers, active=False)
			if X_val is not None and Y_val is not None:
				self.test(models, X_val, Y_val, query_infos_val)
		for iter in range(self.active_iters):
			select_indices = self.active_test(models, X_test, Y_test)
			X_train, Y_train, X_test, Y_test = self.merge_data(select_indices, X_train, Y_train, X_test, Y_test)
			print("The {}-th active Learning: # Training Data {}".format(iter, X_train.shape[0]))
			models, optimizers, schedulers = self.train(models, optimizers, X_train, Y_train, schedulers, active= True)
			if X_val is not None and Y_val is not None:
				self.test(models, X_val, Y_val, query_infos_val)


def main(args):
	#X, Y, all_query_infos = datasets.load_training_data(args)
	X, Y, all_query_infos = schemas.load_training_schema_data(args)
	num_queries, num_feats = X.shape

	print("number of query: {}".format(num_queries))
	X_train, Y_train, query_infos_train, X_test, Y_test, query_infos_test, X_val, Y_val, query_infos_val = \
		train_test_val_split(X, Y, train_frac=0.2, test_frac=0.6, all_query_infos=all_query_infos)
	print(X_train.shape, X_test.shape)
	print(Y_train.shape, Y_test.shape)

	models = []
	# define the model ensemble
	for _ in range(args.ensemble_num):
		models.append(MLPDensityRegressor(in_ch=num_feats, hid_ch=args.num_hid))
	optimizers = [optim.Adam(model.parameters(),
						   lr=args.learning_rate, weight_decay=args.weight_decay) for model in models]
	schedulers = [optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_factor) for optimizer in optimizers]

	active_learner = DeepEnsemble(args)
	active_learner.active_train(models, optimizers, schedulers, X_train, Y_train, X_test, Y_test, X_val, Y_val, query_infos_val, pretrain=True)
	#models, optimizers, schedulers = active_learner.train(models, optimizers, X_train, Y_train, schedulers, active = False)
	#active_learner.test(models, X_test, Y_test, query_infos_test)


if __name__ == "__main__":
	parser = ArgumentParser("Deep Ensemble", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
	parser.add_argument("--chunk_size", default=10, type=int, help="dimension of factorized encoding")
	parser.add_argument("--num_hid", default=512, type=int, help="number of hidden")
	parser.add_argument("--feat_encode", type=str, default='dnn-encoder', help='dnn-encoder,one-hot')

	# Training parameters
	parser.add_argument("--epochs", default=50, type=int)
	parser.add_argument("--learning_rate", default=1e-4, type=float)
	parser.add_argument("--batch_size", default=32, type=int, help="batch size")
	parser.add_argument('--weight_decay', type=float, default=2e-4,
						help='Weight decay (L2 loss on parameters).')
	parser.add_argument('--decay_factor', type=float, default=0.85,
						help='decay rate of (gamma).')
	parser.add_argument('--decay_patience', type=int, default=10,
						help='num of epoches for one lr decay.')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='Disables CUDA training.')

	# Card Classification task
	parser.add_argument("--biased_sample", default=True, type=bool,
						help="Enable Biased sampling for test set selection")
	parser.add_argument('--active_iters', type=int, default=3,
						help='Num of iterators of active learning.')
	parser.add_argument('--budget', type=int, default=1000,
						help='Selected Queries budget Per Iteration.')
	parser.add_argument('--active_epochs', type=int, default=50,
						help='Training Epochs for per iteration active learner.')
	parser.add_argument('--ensemble_num', type=int, default=5,
						help='number of ensemble models for active learning.')

	# input dir
	#parser.add_argument("--relations", type=str, default='yelp-review,yelp-user')  # 'yelp-user'
	#parser.add_argument("--names", type=str, default='review,user')
	#parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/join_review_user_100_2')

	parser.add_argument("--relations", type=str, default='forest')
	parser.add_argument("--names", type=str, default='forest')
	parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/forest_data')

	parser.add_argument("--schema_name", type=str, default='tpch', help='yelp, tpcds, tpch')

	# parser.add_argument("--query_path", type=str, default="/home/kfzhao/PycharmProjects/NNGP/queryset/join_business_review_user_10_data_centric")
	#parser.add_argument("--query_path", type=str,  default="/home/kfzhao/PycharmProjects/NNGP/queryset/join_store_sales_store_item_customer_promotion_10_data_centric_422")
	parser.add_argument("--query_path", type=str,  default="/home/kfzhao/PycharmProjects/NNGP/queryset/join_lineitem_part_orders_supplier_10_data_centric_427")


	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	args.device = torch.device('cuda' if args.cuda else 'cpu')
	relations = args.relations.split(',')
	args.join_query = True if len(relations) > 1 else False
	print(args)
	main(args)