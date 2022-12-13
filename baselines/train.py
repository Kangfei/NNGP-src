from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from baselines.layers import MSCN, MSCNJoin, TreeLSTM
import datasets
import datetime
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, accuracy_score
import math
from util import PredictionStatistics
from baselines.dataset import train_test_val_split, MSCNDataset
import torch

pred_stat = PredictionStatistics()

def test(args, model, criterion, X_test, Y_test, query_infos_test= None):
	model.eval()
	outputs = None
	test_dataset = MSCNDataset(X_test, Y_test, args.join_query, args.max_classes)
	test_loader = DataLoader(test_dataset, batch_size=5000, shuffle=False)
	total_loss = 0.0
	start = datetime.datetime.now()
	for i, data in enumerate(test_loader):
		output = None
		if args.join_query:
			left_pred_x, right_pred_x, join_x, y, _ = data
			if args.cuda:
				left_pred_x, right_pred_x, join_x, y = left_pred_x.cuda(), right_pred_x.cuda(), join_x.cuda(), y.cuda()
			output = model(left_pred_x, right_pred_x, join_x)
		else:
			pred_x, y, _ = data
			if args.cuda:
				pred_x, y = pred_x.cuda(), y.cuda()
			output = model(pred_x)
		loss = criterion(output, y)
		total_loss += loss.item()
		outputs = output if outputs is None else torch.cat([outputs, output], dim=0)
	end = datetime.datetime.now()
	duration = (end - start).total_seconds()
	print("MSCN/TLSTM Total Inference time={} seconds".format(duration))
	print("Test MSE Loss={:.4f}".format(total_loss))
	errors = outputs.cpu().detach().numpy() - Y_test
	pred_stat.get_prediction_details(errors, query_infos_test, partition_keys="num_predicates")



def train(args, model, optimizer, criterion, X_train, Y_train, scheduler = None):
	if args.cuda:
		model.to(args.device)
	train_dataset = MSCNDataset(X_train, Y_train, args.join_query, args.max_classes)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

	start = datetime.datetime.now()
	for epoch in range(args.epochs):
		total_loss = 0.0
		model.train()
		for i, data in enumerate(train_loader):
			optimizer.zero_grad()
			output = None
			if args.join_query:
				left_pred_x, right_pred_x, join_x, y, _ = data
				if args.cuda:
					left_pred_x, right_pred_x, join_x, y = left_pred_x.cuda(), right_pred_x.cuda(), join_x.cuda(), y.cuda()
				output = model(left_pred_x, right_pred_x, join_x)
			else:
				pred_x, y, _ = data
				if args.cuda:
					pred_x, y = pred_x.cuda(), y.cuda()
				output = model(pred_x)
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
	# perform test
	return model

def main(args):
	X, Y, all_query_infos = datasets.load_training_data(args)
	num_queries = len(X)
	pred_feat, join_feat = (X[0][0].shape[1], X[0][2].shape[1]) if args.join_query else (X[0].shape[1], 0)
	print("number of query: {}, pred feat dim: {}, join feat dim; {}".format(num_queries, pred_feat, join_feat))
	X_train, Y_train, query_infos_train, X_test, Y_test, query_infos_test, X_val, Y_val, query_infos_val = \
		train_test_val_split(X, Y, train_frac=0.6, test_frac=0.2, all_query_infos=all_query_infos)
	print(len(X_train), len(X_test))
	print(Y_train.shape, Y_test.shape)
	pred_num_hid = args.pred_num_hid
	join_num_hid = args.join_num_hid
	pred_num_out = args.pred_num_out
	join_num_out = args.join_num_out
	mlp_num_hid = args.mlp_num_hid

	if not args.join_query:
		model = MSCN(pred_feat, pred_num_hid, pred_num_out, mlp_num_hid)
	elif args.model_type == 'MSCN':
		model = MSCNJoin(pred_feat, pred_num_hid, pred_num_out, join_feat, join_num_hid, join_num_out, mlp_num_hid)
	else:
		model = TreeLSTM(pred_feat, pred_num_hid, pred_num_out, join_feat, join_num_hid, join_num_out, mlp_num_hid)
	print(model)
	criterion = torch.nn.MSELoss()
	optimizer = optim.Adam(model.parameters(),
						   lr=args.learning_rate, weight_decay=args.weight_decay)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_factor)
	model = train(args, model, optimizer, criterion, X_train, Y_train, scheduler)
	test(args, model, criterion, X_test, Y_test, query_infos_test)


if __name__ == "__main__":
	parser = ArgumentParser("DNN estimator", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
	parser.add_argument("--chunk_size", default=10, type=int, help="dimension of factorized encoding")
	parser.add_argument("--model_type", type=str, default='TLSTM', help='MSCN, TLSTM')
	parser.add_argument("--mlp_num_hid", default=256, type=int, help="number of hidden")
	parser.add_argument("--pred_num_hid", default=64, type=int, help="number of hidden")
	parser.add_argument("--join_num_hid", default=64, type=int, help="number of hidden")
	parser.add_argument("--pred_num_out", default=64, type=int, help="number of hidden")
	parser.add_argument("--join_num_out", default=64, type=int, help="number of hidden")
	parser.add_argument("--feat_encode", type=str, default='one-hot', help='dnn-encoder,one-hot')

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
	#parser.add_argument("--relations", type=str, default='forest')
	#parser.add_argument("--names", type=str, default='forest')
	#parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/forest_data')

	parser.add_argument("--relations", type=str, default='higgs')
	parser.add_argument("--names", type=str, default='higgs')
	parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/higgs')

	#parser.add_argument("--relations", type=str, default='yelp-review,yelp-user') #'yelp-user'
	#parser.add_argument("--names", type=str, default='review,user')
	#parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/join_review_user_100_2')

	#parser.add_argument("--relations", type=str, default='forest,forest')  # 'forest,forest'
	#parser.add_argument("--names", type=str, default='forest1,forest2')
	#parser.add_argument("--query_path", type=str, default='/home/kfzhao/PycharmProjects/NNGP/queryset/join_forest1_forest2')


	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	args.device = torch.device('cuda' if args.cuda else 'cpu')
	relations = args.relations.split(',')
	args.join_query = True if len(relations) > 1 else False
	print(args)
	main(args)