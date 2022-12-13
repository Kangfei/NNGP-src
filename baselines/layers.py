import torch
import torch.nn as nn
import torch.nn.functional as F


class SetConvolution(nn.Module):
	def __init__(self, in_ch, hid_ch, out_ch, num_layers=2, pool_type ='mean'):
		super(SetConvolution, self).__init__()
		self.num_layers = num_layers
		self.pool_type = pool_type
		self.layers = nn.ModuleList()
		for i in range(self.num_layers):
			hid_input_ch = in_ch if i == 0 else hid_ch
			hid_output_ch = out_ch if i == self.num_layers - 1 else hid_ch
			self.layers.append(nn.Linear(hid_input_ch, hid_output_ch))

	def forward(self, x):
		for i in range(self.num_layers):
			x = self.layers[i](x)
			x = F.relu(x)
		if self.pool_type == 'mean':
			x = torch.mean(x, dim= 1)
		elif self.pool_type == 'min':
			x , _ = torch.min(x, dim= 1) # return (val, index)
		else:
			assert False, "Unsupported pool type in set convolution!"
		return x

class MLP(nn.Module):
	def __init__(self, in_ch, hid_ch, out_ch):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(in_ch, hid_ch)
		self.fc2 = nn.Linear(hid_ch, out_ch)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		return self.fc2(x)

class MSCNJoin(nn.Module):
	def __init__(self, pred_in_ch, pred_hid_ch, pred_out_ch, join_in_ch, join_hid_ch, join_out_ch, mlp_hid_ch):
		super(MSCNJoin, self).__init__()
		self.pred_set_conv = SetConvolution(pred_in_ch, pred_hid_ch, pred_out_ch, num_layers=2)
		self.join_set_conv = SetConvolution(join_in_ch, join_hid_ch, join_out_ch, num_layers=2)
		self.mlp = MLP(in_ch=pred_out_ch + join_out_ch, hid_ch=mlp_hid_ch, out_ch= 1)

	def forward(self, left_pred_x, right_pred_x, join_x):
		#print(left_pred_x.shape, right_pred_x.shape, join_x.shape)
		pred_x = torch.cat([left_pred_x, right_pred_x], dim=1)
		#print("pred_x:", pred_x.shape)
		pred_x = self.pred_set_conv(pred_x)
		join_x = self.join_set_conv(join_x)
		x = torch.cat([pred_x, join_x], dim=1)
		x = self.mlp(x)
		return x

class MSCNMultiJoin(nn.Module):
	def __init__(self, table_in_ch, table_hid_ch, table_out_ch, pred_in_ch, pred_hid_ch, pred_out_ch, join_in_ch, join_hid_ch, join_out_ch, mlp_hid_ch):
		super(MSCNMultiJoin, self).__init__()
		self.table_set_cov = SetConvolution(table_in_ch, table_hid_ch, table_out_ch, num_layers=2)
		self.pred_set_conv = SetConvolution(pred_in_ch, pred_hid_ch, pred_out_ch, num_layers=2)
		self.join_set_conv = SetConvolution(join_in_ch, join_hid_ch, join_out_ch, num_layers=2)
		self.mlp = MLP(in_ch=table_out_ch + pred_out_ch + join_out_ch, hid_ch=mlp_hid_ch, out_ch=1)

	def forward(self, table_x, pred_x, join_x):
		#print(table_x.shape, pred_x.shape, join_x.shape)
		table_x = self.table_set_cov(table_x)
		pred_x = self.pred_set_conv(pred_x)
		join_x = self.join_set_conv(join_x)
		x = torch.cat([table_x, pred_x, join_x], dim= 1)
		x = self.mlp(x)
		return x


class MSCN(nn.Module):
	def __init__(self, pred_in_ch, pred_hid_ch, pred_out_ch, mlp_hid_ch):
		super(MSCN, self).__init__()
		self.pred_set_conv = SetConvolution(pred_in_ch, pred_hid_ch, pred_out_ch, num_layers=2)
		self.mlp = MLP(in_ch=pred_out_ch, hid_ch=mlp_hid_ch, out_ch= 1)

	def forward(self, pred_x):
		pred_x = self.pred_set_conv(pred_x)
		x = self.mlp(pred_x)
		return x


class TreeLSTM(nn.Module):
	def __init__(self, pred_in_ch, pred_hid_ch, pred_out_ch, join_in_ch, join_hid_ch, join_out_ch, mlp_hid_ch):
		super(TreeLSTM, self).__init__()
		self.pred_set_conv = SetConvolution(pred_in_ch, pred_hid_ch, pred_out_ch, num_layers= 2, pool_type='min')
		self.join_set_conv = SetConvolution(join_in_ch, join_hid_ch, join_out_ch, num_layers=2)
		self.lstm = nn.LSTM(input_size=pred_out_ch + join_out_ch, hidden_size=mlp_hid_ch)
		self.mlp = MLP(in_ch=self.lstm.hidden_size, hid_ch= mlp_hid_ch, out_ch = 1)

	def forward(self, left_pred_x, right_pred_x, join_x):
		left_pred_x = self.pred_set_conv(left_pred_x)
		right_pred_x = self.pred_set_conv(right_pred_x)
		pred_x = (left_pred_x + right_pred_x) / 2.0
		#print(pred_x.shape, join_x.shape)
		join_x = self.join_set_conv(join_x)
		x = torch.cat([pred_x, join_x], dim = 1)
		x = x.unsqueeze(dim= 0)
		x, (_, _) = self.lstm(x)
		x = self.mlp(x)
		x = x.squeeze()
		return x



class TreeLSTMMulitJoin(nn.Module):
	def __init__(self, op_feat, op_out_ch, meta_feat, meta_out_ch, pred_in_ch, pred_hid_ch, pred_out_ch, lstm_hid_ch, mlp_hid_ch):
		super(TreeLSTMMulitJoin, self).__init__()
		self.op_nn = nn.Sequential(nn.Linear(in_features=op_feat, out_features=op_out_ch), nn.ReLU())
		self.meta_nn = nn.Sequential(nn.Linear(in_features=meta_feat, out_features=meta_out_ch), nn.ReLU())
		self.pred_set_cov = SetConvolution(pred_in_ch, pred_hid_ch, pred_out_ch, num_layers=2, pool_type='min')
		lstm_in_ch = lstm_hid_ch + op_out_ch + meta_out_ch + pred_out_ch
		self.pad_zeros = torch.zeros((1, lstm_hid_ch))

		self.lstm = nn.LSTM(input_size=lstm_in_ch, hidden_size=lstm_hid_ch)
		self.mlp = MLP(in_ch=self.lstm.hidden_size, hid_ch=mlp_hid_ch, out_ch=1)

	def forward(self, root):
		plan_x, (_, _) = self.recursive_forward(root)
		x = self.mlp(plan_x)
		x = x.squeeze(dim=0)
		return x


	def recursive_forward(self, root):
		# op_features : [1, op_feat]
		# meta_features : [1, meta_feat]
		# pred_x : [1, num_pred, pred_feat]
		#print(root.op_features.shape)
		op_x = self.op_nn(root.op_features) # [1, op_out_ch]
		#print(op_x.shape)
		meta_x = self.meta_nn(root.meta_features) # [1, meta_out_ch]
		#print(root.pred_features.shape)
		pred_x = self.pred_set_cov(root.pred_features) # [1, pred_out_ch]
		x = torch.cat([op_x, meta_x, pred_x], dim=1) # [1, op_out_ch + meta_out_ch, pred_out_ch]
		if root.level == 0:
			x = torch.cat([self.pad_zeros, x], dim=1) # [1, lstm_hid_ch + op_out_ch + meta_out_ch, pred_out_ch]
			x = x.unsqueeze(dim=0)
			return self.lstm(x)
		#
		l, (_, _) = self.recursive_forward(root.children[0])
		r, (_, _) = self.recursive_forward(root.children[1])
		l, r = l.squeeze(dim= 0), r.squeeze(dim= 0) # [1, lstm_hid_ch]
		x = torch.cat([ (l + r) / 2, x], dim=1)
		x = x.unsqueeze(dim=0) # [1, lstm_hid_ch + op_out_ch + meta_out_ch, pred_out_ch]
		return self.lstm(x)
