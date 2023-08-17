'''
This file contains the code to convert the csv files into graph data for node classification using GCN.
Usage: python build_graph.py
'''
import pickle
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
import pickle as pkl
from itertools import combinations
import torch_geometric.data as make_dataset
from labeling_functions import apply_lfs

def build_graph(df, level_of_analysis='LSH label'):
	# function builds graph based on micro and meta cluster labels
	num_micro = df[level_of_analysis].nunique()
	# clusters = df[level_of_analysis].unique()
	clus_ind_map = dict(zip(df[level_of_analysis].unique(),range(num_micro)))
	micro_to_meta_map = np.zeros([num_micro, num_micro])

	for clus, grp in tqdm(df.groupby("Meta label")):
		clusters = grp[level_of_analysis].unique()
		for e1,e2 in tqdm(combinations(clusters,2)):
			micro_to_meta_map[clus_ind_map[e1]][clus_ind_map[e2]] += 1

	nx_graph = nx.Graph(micro_to_meta_map).to_undirected()
	return nx_graph


def find_meta_clusters(df, level_of_analysis='LSH label'):
	# function to find meta-cluster labels
	print("Finding meta clusters...\n")
	num_micro = df[level_of_analysis].nunique()
	clus_ind_map = dict(zip(df[level_of_analysis].unique(),range(num_micro)))
	micro_to_meta_map = np.zeros([num_micro, num_micro])
	
	phone_col = "phone"
	p = df.dropna(subset=[phone_col])
	for id, grp in tqdm(p.groupby(phone_col)):
	# p = df.dropna(subset=['phone'])
	# for id, grp in tqdm(p.groupby('phone')):
	    clusters = grp[level_of_analysis].unique()
	    pairs = combinations(clusters, 2)
	    for e1, e2 in pairs:
	        micro_to_meta_map[clus_ind_map[e1]][clus_ind_map[e2]] += 1
	
	if 'image_id' in df.columns:
		p = df.dropna(subset=['image_id'])  
		for id, grp in tqdm(p.groupby('image_id')):
		    clusters = grp[level_of_analysis].unique()
		    pairs = combinations(clusters, 2)
		    for e1, e2 in pairs:
		        micro_to_meta_map[clus_ind_map[e1]][clus_ind_map[e2]] += 1

	if 'email' in df.columns:
		p = df.dropna(subset=['email'])
		for id, grp in tqdm(p.groupby('email')):
		    clusters = grp[level_of_analysis].unique()
		    pairs = combinations(clusters, 2)
		    for e1, e2 in pairs:
		        micro_to_meta_map[clus_ind_map[e1]][clus_ind_map[e2]] += 1

	if 'social' in df.columns:
		p = df.dropna(subset=['social'])
		for id, grp in tqdm(p.groupby('social')):
		    clusters = grp[level_of_analysis].unique()
		    pairs = combinations(clusters, 2)
		    for e1, e2 in pairs:
		        micro_to_meta_map[clus_ind_map[e1]][clus_ind_map[e2]] += 1

	nx_graph = nx.Graph(micro_to_meta_map).to_directed()
	# finding connected components as meta clusters
	meta_label_map = {}
	clus_counter = 0
	num_comps = 0
	for compo in nx.strongly_connected_components(nx_graph):
	    num_comps += 1
	    for node in list(compo):
	        meta_label_map[node] = clus_counter
	    clus_counter += 1
	df['Meta label'] = df[level_of_analysis].apply(lambda x:meta_label_map[clus_ind_map[x]])
	print("Number of meta clusters = " + str(df['Meta label'].nunique()))
	return df, nx_graph

def preprocess(df, cities):
	# cities = cities[cities.country_id==3]
	df = pd.merge(df, cities, left_on='location', right_on='name', how='left')
	df['ad_id'] = df['id'].copy()
	df.rename(columns={'phone':'phone_num', 'cleaned_text':'description'}, inplace=True)
	return df

def get_weak_labels(data, level_of_analysis='LSH label'):
	# weak_labels = pickle.load(open("marinus_labelled/merged_data_3_class_no_dupl_names_LSH_labels_weak_labels2.pkl",'rb'))
	if 'geolocation' not in data.columns:
		cities = pd.read_csv("ht_datasets/marinus_labelled/cities.csv", index_col=False)
		data = preprocess(data, cities)
	label_mat, spa_clusters, lf_votes = apply_lfs(data, level_of_analysis=level_of_analysis)
	# pkl.dump(lf_votes, open("ht_datasets/synthetic_aws2/lf_votes.pkl",'wb'))
	# lf_votes = pkl.load(open("ht_datasets/synthetic_aws2/lf_votes.pkl",'rb'))	

	clus_ind_map = dict(zip(data[level_of_analysis].unique(),range(data[level_of_analysis].nunique())))
	lambda_mat = np.zeros(shape=[data[level_of_analysis].nunique(), 12])
	for lsh_label, per_class_labels in lf_votes.items():
		# 0 - SPAM
		# 1 - HT
		# 2 - ISW
		row_indx = clus_ind_map[lsh_label]
		col_indx = 0
		for vote in per_class_labels['ht']:
			if vote: # if o/p of LF is true
				lambda_mat[row_indx, col_indx] = 1
			else:
				# randomly choose between abstain and ISW (we are defaulting to ISW class)
				# chosen_label = np.random.choice([-1,2],p=[1,0],size=1)[0]
				chosen_label = -1
				lambda_mat[row_indx, col_indx] = chosen_label # abstaining or ISW
			col_indx += 1
		for vote in per_class_labels['isw']:
			if vote: # if o/p of LF is true
				lambda_mat[row_indx, col_indx] = 2
			else:
				# randomly choose between abstain and ISW (we are defaulting to ISW class)
				# chosen_label = np.random.choice([-1,2],p=[0.8,0.2],size=1)[0]
				lambda_mat[row_indx, col_indx] = -1 # abstaining or ISW
			col_indx += 1
		for vote in per_class_labels['spam']:
			if vote: # if o/p of LF is true
				lambda_mat[row_indx, col_indx] = 0
			else:
				# randomly choose between abstain and ISW (we are defaulting to ISW class)
				# chosen_label = np.random.choice([-1,2],p=[1,0],size=1)[0]
				chosen_label = -1
				lambda_mat[row_indx, col_indx] = chosen_label # abstaining or ISW
			col_indx += 1

	return lambda_mat

def get_labels(df, level_of_analysis='LSH label'):
	clus_ind_map = dict(zip(df[level_of_analysis].unique(),range(df[level_of_analysis].nunique())))
	label_df = df[[level_of_analysis, 'mo_labels']].drop_duplicates().to_numpy()
	label_dict = dict(label_df)
	labels = np.zeros(len(label_dict))
	label_map = {'ht':1, 'spam':0, 'isw':2}
	for lsh_label, class_label in label_dict.items():
		if pd.isna(class_label):
			class_label = 'isw'
		labels[clus_ind_map[lsh_label]] = label_map[class_label]

	return labels

def get_graph(nx_graph, y, feats, df, level_of_analysis='LSH label'):
	edge_index = [[],[]]
	for line in nx.generate_edgelist(nx_graph, data=False):
		edge_index[0].append(int(line.split()[0]))
		edge_index[1].append(int(line.split()[1]))

	data = make_dataset.Data(x=feats, y=y, edge_index=edge_index)
	data.weak_labels = get_weak_labels(df, level_of_analysis=level_of_analysis)
	return data

def modify_feats(feat_df, clus_ind_map):
	cols_to_keep = []
	for col in feat_df.columns:
		if 'Val' not in col and col != 'Num URLs':
			cols_to_keep.append(col)

	feats = feat_df[cols_to_keep].to_numpy()
	modified_feats = np.zeros(shape=[len(feats), len(feats[0])-1])
	for row in feats:
		if row[-1] == -1:
			continue
		indx = clus_ind_map[row[-1]]
		modified_feats[indx] = row[:-1]

	return modified_feats


def get_data_df():

	data_path = sys.argv[1]
	save_path = sys.argv[2]
	data = pd.read_csv(data_path, index_col=False)
	
	level_of_analysis = 'cluster_id'
	get_weak_labels(data, level_of_analysis=level_of_analysis)
	nx_graph = build_graph(data, level_of_analysis=level_of_analysis)
	
	# data, nx_graph = find_meta_clusters(data, level_of_analysis=level_of_analysis)

	feat_path = "".join(x for x in data_path.split(".csv")[0].split("/")[:-1])

	feat_df = pd.read_csv(feat_path, index_col=False)
	
	clus_ind_map = dict(zip(data[level_of_analysis].unique(),range(data[level_of_analysis].nunique())))
	feats = modify_feats(feat_df, clus_ind_map)

	labels = get_labels(data, level_of_analysis=level_of_analysis)

	data_graph = get_graph(nx_graph, labels, feats, data, level_of_analysis=level_of_analysis)
	print(data[level_of_analysis].nunique(), feat_df.shape)

	pickle.dump(data_graph, open(save_path,'wb'))


if __name__ == '__main__':
	get_data_df()

