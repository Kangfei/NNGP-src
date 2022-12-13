from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import schemas
import datasets
import datetime
import torch.optim as optim
from torch.utils.data import DataLoader
from baselines.layers import MSCNMultiJoin, TreeLSTMMulitJoin
import random
from sklearn.metrics import mean_squared_error, accuracy_score
import math
import sys
from util import PredictionStatistics
from baselines.dataset import train_test_val_split, MultiJoinMSCNDataset
import torch

pred_stat = PredictionStatistics()

def test(args, model, criterion, X_test, Y_test, query_infos_test=None):
	model.eval()
	outputs = []
	test_dataset = MultiJoinMSCNDataset(X_test, Y_test, args.max_classes)
	test_loader = DataLoader(test_dataset, batch_size=5000, shuffle=False)
	total_loss = 0.0
	start = datetime.datetime.now()
	for i, (table_x, pred_x, join_x, y, _) in enumerate(test_loader):
		if args.cuda:
			table_x, pred_x, join_x, y = table_x.cuda(), pred_x.cuda(), join_x.cuda(), y.cuda()
		output = model(table_x, pred_x, join_x)
		loss = criterion(output, y)
		total_loss += loss.item()
		outputs.append(output)
	end = datetime.datetime.now()
	duration = (end - start).total_seconds()
	print('MSCN Test in %s seconds.' % duration)
	print("Test MSE Loss={:.4f}".format(total_loss))
	outputs = torch.cat(outputs, dim=0)
	errors = outputs.cpu().detach().numpy() - Y_test
	pred_stat.get_prediction_details(errors, query_infos_test, partition_keys='num_table')


def train(args, model, optimizer, criterion, X_train, Y_train, scheduler = None):
	if args.cuda:
		model.to(args.device)
	train_dataset = MultiJoinMSCNDataset(X_train, Y_train, args.max_classes)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

	start = datetime.datetime.now()
	for epoch in range(args.epochs):
		total_loss = 0.0
		model.train()
		for i, (table_x, pred_x, join_x, y, _) in enumerate(train_loader):
			optimizer.zero_grad()
			if args.cuda:
				table_x, pred_x, join_x, y = table_x.cuda(), pred_x.cuda(), join_x.cuda(), y.cuda()
			output = model(table_x, pred_x, join_x)
			loss = criterion(output, y)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		print("{}-th Epochs: Train MSE Loss={:.4f}".format(epoch, total_loss))
		if scheduler is not None and (epoch + 1) % args.decay_patience == 0:
			scheduler.step()
	end = datetime.datetime.now()
	duration = (end - start).total_seconds()
	print('MSCN Training in %s seconds.' % duration)
	return model

def test_lstm(args, model, criterion, X_test, Y_test, query_infos_test=None):
	model.eval()
	outputs = []
	for root in X_test:
		root.recursive_to_torch_tensor(args.cuda)
	Y_test = [torch.FloatTensor(y) for y in Y_test.tolist()]
	total_loss = 0.0
	start = datetime.datetime.now()
	for i, (root, y) in enumerate(zip(X_test, Y_test)):
		if args.cuda:
			y = y.cuda()
		output = model(root)
		loss = criterion(output, y)
		total_loss += loss
		outputs.append(output)
	end = datetime.datetime.now()
	duration = (end - start).total_seconds()
	print('Tree LSTM Test in %s seconds.' % duration)
	print("Test MSE Loss={:.4f}".format(total_loss))
	outputs = torch.cat(outputs, dim=0)
	errors = outputs.cpu().detach().numpy() - Y_test
	pred_stat.get_prediction_details(errors, query_infos_test, partition_keys='num_table')


def train_lstm(args, model, optimizer, criterion, X_train, Y_train, scheduler = None):
	if args.cuda:
		model.to(args.device)
	for root in X_train:
		root.recursive_to_torch_tensor(args.cuda)

	Y_train = [ torch.FloatTensor(y) for y in Y_train.tolist()]
	X_Y_train = list(zip(X_train, Y_train))
	start = datetime.datetime.now()
	for epoch in range(args.epochs):
		total_loss = 0.0
		model.train()
		random.shuffle(X_Y_train)
		for i, (root, y) in enumerate(X_Y_train):
			if args.cuda:
				y = y.cuda()
			output = model(root)

			loss = criterion(output, y)
			total_loss += loss.item()
			loss.backward()
			if (i + 1) % args.batch_size == 0:
				optimizer.step()
				optimizer.zero_grad()
		print("{}-th Epochs: Train MSE Loss={:.4f}".format(epoch, total_loss))
		if scheduler is not None and (epoch + 1) % args.decay_patience == 0:
			scheduler.step()
	end = datetime.datetime.now()
	duration = (end - start).total_seconds()
	print('Tree LSTM Training in %s seconds.' % duration)
	return model


def main(args):
	table_num_hid = args.table_num_hid
	pred_num_hid = args.pred_num_hid
	join_num_hid = args.join_num_hid
	table_num_out = args.table_num_out
	meta_out_ch = args.meta_num_out
	op_out_ch = args.op_num_out
	pred_num_out = args.pred_num_out
	join_num_out = args.join_num_out
	mlp_num_hid = args.mlp_num_hid
	lstm_num_hid = args.lstm_num_hid


	X, Y, all_query_infos = schemas.load_training_schema_data(args)


	X_train, Y_train, query_infos_train, X_test, Y_test, query_infos_test, X_val, Y_val, query_infos_val = \
		train_test_val_split(X, Y, train_frac=0.6, test_frac=0.2, all_query_infos=all_query_infos)

	num_queries = len(X)
	if args.model_type == 'MSCN':
		table_feat, pred_feat, join_feat = X[0][0].shape[1],  X[0][1].shape[1],  X[0][2].shape[1]
		print("number of query: {}, table feat dim: {}, pred feat dim: {}, join feat dim; {}" \
			  .format(num_queries, table_feat, pred_feat,join_feat))
		model = MSCNMultiJoin(table_feat, table_num_hid, table_num_out, pred_feat, pred_num_hid, pred_num_out,
							  join_feat, join_num_hid, join_num_out, mlp_num_hid)
		print(model)
		criterion = torch.nn.MSELoss()
		optimizer = optim.Adam(model.parameters(),
							   lr=args.learning_rate, weight_decay=args.weight_decay)
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_factor)
		model = train(args, model, optimizer, criterion, X_train, Y_train, scheduler)
		test(args, model, criterion, X_test, Y_test, query_infos_test)

	else:
		meta_feat, pred_feat, op_feat = X[0].meta_features.shape[0], X[0].pred_features.shape[1], X[0].op_features.shape[0]
		print("number of query: {}, meta feat dim: {}, pred feat dim: {}, op feat dim; {}" \
			  .format(num_queries, meta_feat, pred_feat, op_feat))
		model = TreeLSTMMulitJoin(op_feat, op_out_ch, meta_feat, meta_out_ch, pred_feat, pred_num_hid, pred_num_out, lstm_num_hid, mlp_num_hid)
		print(model)
		criterion = torch.nn.MSELoss()
		optimizer = optim.Adam(model.parameters(),
							   lr=args.learning_rate, weight_decay=args.weight_decay)
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_factor)
		model = train_lstm(args, model, optimizer, criterion, X_train, Y_train, scheduler)
		test_lstm(args, model, criterion, X_test, Y_test, query_infos_test)



if __name__ == "__main__":
	parser = ArgumentParser("DNN estimator", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
	parser.add_argument("--chunk_size", default=10, type=int, help="dimension of factorized encoding")
	parser.add_argument("--model_type", type=str, default='MSCN', help='MSCN, TLSTM')
	parser.add_argument("--mlp_num_hid", default=256, type=int, help="number of hidden")
	parser.add_argument("--lstm_num_hid", default=256, type=int, help="number of hidden")
	parser.add_argument("--table_num_hid", default=64, type=int, help="number of hidden")
	parser.add_argument("--pred_num_hid", default=64, type=int, help="number of hidden")
	parser.add_argument("--join_num_hid", default=64, type=int, help="number of hidden")
	parser.add_argument("--table_num_out", default=64, type=int, help="number of hidden")
	parser.add_argument("--op_num_out", default=64, type=int, help="number of hidden")
	parser.add_argument("--meta_num_out", default=64, type=int, help="number of hidden")
	parser.add_argument("--pred_num_out", default=64, type=int, help="number of hidden")
	parser.add_argument("--join_num_out", default=64, type=int, help="number of hidden")
	parser.add_argument("--feat_encode", type=str, default='one-hot', help='dnn-encoder,one-hot')

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

	parser.add_argument("--multi_task", default=True, type=bool,
						help="enable/disable card classification task.")
	parser.add_argument("--max_classes", default=10, type=int,
						help="number classes for the card classification task.")
	parser.add_argument('--coeff', type=float, default=0.5,
						help='coefficient for the classification loss.')
	parser.add_argument("--uncertainty", default="consist", type=str,
						help="The uncertainty type")  # entropy, margin, confident, consist, random are tested

	parser.add_argument("--schema_name", type=str, default='tpcds', help='yelp, tpcds, tpch')

	#parser.add_argument("--query_path", type=str, default="/home/kfzhao/PycharmProjects/NNGP/queryset/join_business_review_user_10_data_centric")
	parser.add_argument("--query_path", type=str,  default="/home/kfzhao/PycharmProjects/NNGP/queryset/join_store_sales_store_item_customer_promotion_10_data_centric_427")
	#parser.add_argument("--query_path", type=str,  default="/home/kfzhao/PycharmProjects/NNGP/queryset/join_lineitem_part_orders_supplier_10_data_centric_427")


	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	args.device = torch.device('cuda' if args.cuda else 'cpu')
	print(args)
	main(args)