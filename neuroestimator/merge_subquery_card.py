import collections
import numpy as np
import math

PredInfo = collections.namedtuple('PredInfo', ['query_str', 'nngp_card', 'nngp_std', 'pg_card', 'true_card'])


def load_card_csv(card_csv_path):
	all_pred_info = list()
	reader = open(card_csv_path, 'r')
	next(reader)
	all_rows = reader.readlines()
	i = 0
	while i < len(all_rows):

		row = all_rows[i]
		i += 1
		row = row.split(";")
		# print(row)
		query_str, nngp_card, nngp_std, pg_card, mix_card, true_card = row[0], float(row[1]), float(row[2]), float( row[3]), float(row[4]), float(row[5])
		if nngp_card <= 0:
			continue
		pred_info = PredInfo(query_str=query_str, nngp_card=nngp_card, nngp_std=nngp_std, pg_card=pg_card, true_card=true_card)
		all_pred_info.append(pred_info)
	print(len(all_pred_info))
	return all_pred_info


def load_subquery_str(subquery_file: str):
	reader = open(subquery_file, 'r')
	# next(reader)
	all_rows = reader.readlines()
	print(len(all_rows))
	return all_rows


def merge_query_res(all_pred_info, all_rows):
	l1 = len(all_pred_info)
	l2 = len(all_rows)
	if l1 < l2:
	    all_rows = all_rows[0 : l1]
	else:
		all_pred_info = all_pred_info[0 : l2]
	assert len(all_pred_info) == len(all_rows), "Inconsistant card/query number!"
	all_line = list()
	max_q_error = 0.0
	ind =  0
	for row, pred_info in zip(all_rows, all_pred_info):
		ind += 1
		if row[0] == '#':
			continue
		row = row.split('@')
		true_card = int(float(row[-1]))
		csv_true_card = int(float(pred_info.true_card))
		assert true_card == csv_true_card or csv_true_card <= 0, "Inconsistant ture card. line no = {ln}, {c1}, {c2}. {s1}, {s2} ".format(ln = ind, c1 = str(true_card), c2 = str(csv_true_card), s1 = row, s2 = pred_info)
		nngp_card, nngp_std = pred_info.nngp_card, pred_info.nngp_std
		coef_var = nngp_std / math.log(nngp_card, 2.0)
		nngp_q_error = max(nngp_card / true_card, true_card / nngp_card)
		max_q_error = max(max_q_error, nngp_q_error)
		row = row[: len(row) - 1] + [str(int(true_card)), str(nngp_q_error), str(coef_var)]
		line = '@'.join(row)
		all_line.append(line)
	print(len(all_line))
	print(max_q_error)

	with open("/home/kfzhao/lirui/data/join_query_aux.txt", 'w') as out_file:
		for line in all_line:
			# line format
			# query str + '@' + true_card + '@' + nngp_q_error + '@' coef_var
			out_file.write(line + '\n')
		out_file.close()


if __name__ == "__main__":
	card_csv_path = "/home/kfzhao/debugpsql/all_subqueries_join_6/card.csv"
	subquery_file = "/home/kfzhao/debugpsql/chosen_subqueries_join_6/queries.txt"
	all_rows = load_subquery_str(subquery_file)
	all_pred_info = load_card_csv(card_csv_path)
	merge_query_res(all_pred_info, all_rows)



