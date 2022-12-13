import torch
import numpy as np
import random
import schemas
import datetime
import datasets
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from scipy.stats import  entropy
from torch.utils.data import DataLoader
from dnn.train import QueryDataset
from dnn import MultiTaskMLP, MCDropoutModel
from util import train_test_val_split
from util import draw_uncertainty, calibration_plot, PredictionStatistics
import torch.optim as optim

class ActiveLearner(object):
	def __init__(self, args):
		self.args = args
		self.budget = args.budget
		self.batch_size = args.batch_size
		self.max_classes = args.max_classes
		self.uncertainty = args.uncertainty
		self.active_iters = args.active_iters
		self.active_epochs = args.active_epochs
		self.biased_sample = args.biased_sample
		self.pred_stat = PredictionStatistics()

	def train(self, model, criterion, criterion_cal,
			  optimizer, X_train, Y_train, scheduler=None, active = False):
		if self.args.cuda:
			model.to(self.args.device)
		epochs = self.active_epochs if active else self.args.epochs

		train_dataset = QueryDataset(X_train, Y_train, self.max_classes)
		train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
		start = datetime.datetime.now()
		for epoch in range(epochs):
			total_loss = 0.0
			model.train()
			for i, (X, Y, label) in enumerate(train_loader):
				if self.args.cuda:
					X, Y, label = X.cuda(), Y.cuda(), label.cuda()

				optimizer.zero_grad()
				if self.args.model_type == "DNN":
					output, output_cla = model(X)
					loss = criterion(output, Y) + self.args.coeff * criterion_cal(output_cla, label)
				else: # self.args.model_type = "MCDropout":
					mu, sigma = model(X)
					#loss = model.loss(mu, Y, sigma)
					#print(mu, sigma, loss)
					loss = criterion(mu, Y)

				loss.backward()
				optimizer.step()
				total_loss += loss.item()
			print("{}-th Epochs: Train MSE Loss={:.4f}".format(epoch, total_loss))
			if scheduler is not None and (epoch + 1) % self.args.decay_patience == 0:
				scheduler.step()

		end = datetime.datetime.now()
		elapse_time = (end - start).total_seconds()
		print("Training time: {:.4f}s".format(elapse_time))
		return model

	def test(self, model, criterion, X_val, Y_val, query_infos_val):
		if self.args.cuda:
			model.to(self.args.device)
		model.eval()
		eval_dataset = QueryDataset(X_val, Y_val, self.max_classes)
		eval_loader = DataLoader(eval_dataset, batch_size=360, shuffle=False)
		errors = None
		total_loss = 0.0
		for i, (X, Y, _) in enumerate(eval_loader):
			if self.args.cuda:
				X, Y = X.cuda(), Y.cuda()
			if self.args.model_type == "DNN":
				output, _ = model(X)
			else: # self.args.model_type = "MCDropout":
				output, _ = model.predict(X)
			loss = criterion(output, Y)
			total_loss += loss.item()
			error = output - Y
			errors = error if errors is None else torch.cat([errors, error], dim= 0)
		print("Evaluation MSE Loss = {:.4f}".format(total_loss))
		errors = errors.cpu().detach().numpy()
		mse = np.mean(np.power(errors, 2.0))
		print("Test MSE Loss:{}".format(mse))
		self.pred_stat.get_prediction_details(errors, query_infos_val, partition_keys='num_predicates')


	def active_test(self, model, X_test, Y_test):
		assert self.args.multi_task, "Classification Task Disabled, Cannot Deploy Active Learning!"
		if self.args.cuda:
			model.to(self.args.device)
		model.eval()
		test_dataset = QueryDataset(X_test, Y_test, self.max_classes)
		test_loader = DataLoader(test_dataset, batch_size=360, shuffle=False)
		outputs, outputs_cla, sigmas = None, None, None
		for i, (X, _, _) in enumerate(test_loader):
			if self.args.cuda:
				X = X.cuda()
			if self.args.model_type == 'DNN':
				output, output_cla = model(X)
				outputs_cla = output_cla if outputs_cla is None else torch.cat([outputs_cla, output_cla], dim=0)
			else:
				output, sigma = model.predict(X)
				sigmas = sigma if sigmas is None else torch.cat([sigmas, sigma], dim=0)
			outputs = output if outputs is None else torch.cat([outputs, output], dim=0)
		if 	self.args.model_type == 'DNN':
			uncertainties = self.compute_uncertainty(outputs_cla, outputs)
		else:
			if self.args.cuda:
				outputs, sigmas = outputs.cpu(), sigmas.cpu()
			outputs, sigmas = outputs.detach().numpy(), sigmas.detach().numpy()
			uncertainties = sigmas / np.max(outputs, 0)
		uncertainties = uncertainties.ravel()
		uncertainties = uncertainties / np.sum(uncertainties)
		num_selected = self.budget if uncertainties.shape[0] > self.budget else uncertainties.shape[0]

		indices = np.random.choice(a=uncertainties.shape[0], size=num_selected, replace=False,
								   p=uncertainties) \
			if self.biased_sample else np.argsort(uncertainties)[- num_selected:]

		return indices


	def compute_uncertainty(self, output_cal, output):
		output_cal, output = output_cal.squeeze(), output.squeeze()
		output_cal = torch.exp(output_cal) # transform output of log_softmax to probability
		if self.args.cuda:
			output_cal = output_cal.cpu()
			output = output.cpu()
		output = output.detach().numpy()
		output_cal = output_cal.detach().numpy()
		print(output.shape, output_cal.shape)
		if self.uncertainty == "entropy":
			return entropy(output_cal, axis=-1)
		elif self.uncertainty == "confident":
			return 1.0 - np.max(output_cal, axis=-1)
		elif self.uncertainty == "margin":
			output_cal = np.sort(output_cal)
			return output_cal[:, -1] - output_cal[:, -2]
		elif self.uncertainty == "random":
			return np.random.rand(output.shape[0])
		elif self.uncertainty == "consist":
			reg_mag = np.ceil( np.log10( np.power(2.0, output)))
			cla_mag = np.argmax(output_cal, axis=-1)
			return np.power((reg_mag - cla_mag), 2)
		else:
			assert False, "Unsupported uncertainty function!"


	def merge_data(self, select_indices, X_train, Y_train, X_test, Y_test):
		X_delta, Y_delta = X_test[select_indices], Y_test[select_indices]
		X_train_new = np.vstack((X_train, X_delta))
		Y_train_new = np.vstack((Y_train, Y_delta))
		num_test = X_test.shape[0]
		indices = np.array(list(range(num_test)))
		keep_indices = np.setdiff1d(indices, np.asarray(select_indices))
		X_test_new, Y_test_new = X_test[keep_indices], Y_test[keep_indices]
		return X_train_new, Y_train_new, X_test_new, Y_test_new


	def active_train(self, model, criterion, criterion_cla, optimizer,
					 X_train, Y_train, X_test, Y_test, X_val = None, Y_val= None, query_infos_val= None, scheduler=None, pretrain = True):

		if pretrain:
			model = self.train(model, criterion, criterion_cla, optimizer, X_train, Y_train, scheduler= scheduler, active= False)
			if X_val is not None and Y_val is not None:
				self.test(model, criterion, X_val, Y_val, query_infos_val)
		for iter in range(self.active_iters):
			selected_indices = self.active_test(model, X_test, Y_test)
			X_train, Y_train, X_test, Y_test = self.merge_data(selected_indices, X_train, Y_train, X_test, Y_test)
			print("The {}-th active Learning: # Training Data {}".format(iter, X_train.shape[0]))
			model = self.train(model, criterion, criterion_cla, optimizer, X_train, Y_train, scheduler=scheduler, active= True)
			if X_val is not None and Y_val is not None:
				self.test(model, criterion, X_val, Y_val, query_infos_val)


def main(args):
	X, Y, all_query_infos = datasets.load_training_data(args)
	#X, Y, all_query_infos = schemas.load_training_schema_data(args)
	num_queries, num_feats = X.shape
	print("number of query: {}".format(num_queries))
	X_train, Y_train, query_infos_train, X_test, Y_test, query_infos_test, X_val, Y_val, query_infos_val = \
		train_test_val_split(X, Y, train_frac=0.2, test_frac=0.6, all_query_infos=all_query_infos)

	print("X train/test/val shape:", X_train.shape, X_test.shape, X_val.shape)
	print("Y train/test/val shape:", Y_train.shape, Y_test.shape, Y_val.shape)

	# define the model
	if args.model_type == 'DNN':
		model = MultiTaskMLP(in_ch=num_feats, hid_ch=args.num_hid, reg_out_ch=1, cla_out_ch=args.max_classes)
	elif args.model_type == 'MCDropout':
		model = MCDropoutModel(input_dim=num_feats, output_dim=1, hid_dim=args.num_hid)
	else:
		assert False, "Unsupported model type!"
	criterion = torch.nn.MSELoss()
	criterion_cla = torch.nn.NLLLoss()
	optimizer = optim.Adam(model.parameters(),
						   lr=args.learning_rate, weight_decay=args.weight_decay)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_factor)

	active_learner = ActiveLearner(args)
	active_learner.active_train(model, criterion, criterion_cla, optimizer,
								X_train, Y_train, X_test, Y_test, X_val, Y_val, query_infos_val= query_infos_val, scheduler=scheduler, pretrain=True)


if __name__ == "__main__":
	parser = ArgumentParser("DNN estimator", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
	parser.add_argument("--chunk_size", default=10, type=int, help="dimension of factorized encoding")
	parser.add_argument("--num_hid", default=512, type=int, help="number of hidden")
	parser.add_argument("--model_type", type=str, default='MCDropout', help="DNN, MCDropout")
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
	parser.add_argument('--no-cuda', action='store_true', default=True,
						help='Disables CUDA training.')

	# Card Classification task
	parser.add_argument("--multi_task", default=True, type=bool,
						help="enable/disable card classification task.")
	parser.add_argument("--max_classes", default=10, type=int,
						help="number classes for the card classification task.")
	parser.add_argument('--coeff', type=float, default=0.5,
						help='coefficient for the classification loss.')
	# Active Learner settings
	parser.add_argument("--uncertainty", default="margin", type=str,
						help="The uncertainty type") # entropy, margin, confident, consist, random are tested, only applicable for 'model_type' == DNN
	parser.add_argument("--biased_sample", default=True, type=bool,
						help="Enable Biased sampling for test set selection")
	parser.add_argument('--active_iters', type=int, default=3,
						help='Num of iterators of active learning.')
	parser.add_argument('--budget', type=int, default=1000,
						help='Selected Queries budget Per Iteration.')
	parser.add_argument('--active_epochs', type=int, default=50,
						help='Training Epochs for per iteration active learner.')
	parser.add_argument("--relations", type=str, default='higgs')

	# input dir
	#parser.add_argument("--relations", type=str, default='yelp-review,yelp-user') #'reviewer,user'
	#parser.add_argument("--names", type=str, default='review,user')
	#parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/join_review_user_100_2')

	#parser.add_argument("--relations", type=str, default='forest,forest')  # 'forest,forest'
	#parser.add_argument("--names", type=str, default='forest1,forest2')
	#parser.add_argument("--query_path", type=str,	default='/home/kfzhao/PycharmProjects/NNGP/queryset/join_forest1_forest2')

	parser.add_argument("--relations", type=str, default='forest')
	parser.add_argument("--names", type=str, default='forest')
	parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/forest_data')

	#parser.add_argument("--relations", type=str, default='higgs')
	#parser.add_argument("--names", type=str, default='higgs')
	#parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/higgs')

	parser.add_argument("--schema_name", type=str, default='tpch', help='yelp, tpcds, tpch')

	#parser.add_argument("--query_path", type=str, default="/home/kfzhao/PycharmProjects/NNGP/queryset/join_business_review_user_10_data_centric")
	#parser.add_argument("--query_path", type=str,  default="/home/kfzhao/PycharmProjects/NNGP/queryset/join_store_sales_store_item_customer_promotion_10_data_centric_427")
	#parser.add_argument("--query_path", type=str,  default="/home/kfzhao/PycharmProjects/NNGP/queryset/join_lineitem_part_orders_supplier_10_data_centric_427")


	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	args.device = torch.device('cuda' if args.cuda else 'cpu')
	relations = args.relations.split(',')
	args.join_query = True if len(relations) > 1 else False
	print(args)
	main(args)