'''
This file contains code for extracting the node features of the graph data for node classification. 
This code needs to be run prior to converting the data into a graph format.
Usage: python extract_features.py [--level_of_analysis 'LSH label']
'''

import pandas as pd
import numpy as np
import sys
import ast
import os
from tqdm import tqdm

from collections import Counter

from sklearn.metrics import adjusted_rand_score as ari

import pickle as pkl

from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE

import umap
import argparse

from features import find_loc_radii, find_entropy, url_count, get_num_ads_per_week



def get_data_df():
	parser = argparse.ArgumentParser()
	parser.add_argument('--filename', help='Path of the file to be preprocessed', \
		default='data/HT2018_final_trimmed_for_labeling_neat_preprocessed.csv')
	parser.add_argument('--level_of_analysis', \
		help='What level of analysis is required', default='Meta label', choices=['Meta label', 'LSH label', 'ad_id', 'cluster_id'])
	parser.add_argument('--recompute', action='store_true', default='False')
	args = parser.parse_args()
	data = pd.read_csv(args.filename, index_col=False)
	level_of_analysis = args.level_of_analysis
	recompute = args.recompute
	data['id'] = data.index.values

	return data, level_of_analysis, recompute, args.filename

if __name__ == "__main__":

	data, level_of_analysis, recompute, filename = get_data_df()

	path_name = "".join(x for x in filename.split(".csv")[0].split("/")[:-1])
	print(path_name)

	if recompute: # if the feature files have not already been saved, compute them
		# computing features
		cluster_sizes = {}
		phone_count = {}
		loc_count = {}
		loc_radii = {}
		phone_entropy = {}
		num_names = {}
		num_valid_urls = {}
		num_invalid_urls = {}
		num_ads_per_week = {}
		num_urls = {}
		mad_values_loc = {}
		num_social = {}
		num_emails = {}

		final_labels = {}
		print(data.columns)
		for ads in tqdm(data.groupby(level_of_analysis)):
			if 'ad_id' in data.columns:
				cluster_sizes[ads[0]] = ads[1].ad_id.count()
			else:
				cluster_sizes[ads[0]] = ads[1].id.count()
			phone_count[ads[0]] = ads[1].phone.nunique()
			loc_count[ads[0]] = ads[1].location.nunique()
			loc_radii[ads[0]] = find_loc_radii(ads[1].location.drop_duplicates())
			# print("finding entropy..")
			phone_entropy[ads[0]] = find_entropy(ads[1].phone.values)
			num_names[ads[0]] = ads[1].names.count()
			# print("finding urls...")
			num_valid_urls[ads[0]], num_invalid_urls[ads[0]], num_urls[ads[0]] = url_count(ads[1].cleaned_text.unique())
			# print("num ads per week...")
			num_ads_per_week[ads[0]] = get_num_ads_per_week(ads[1], col_name='post_dates')
			num_social[ads[0]] = ads[1].social.nunique()

			num_emails[ads[0]] = ads[1].email.nunique()
			
		# saving the cluster features
		pkl.dump(cluster_sizes, open(path_name+"cluster_sizes.pkl",'wb'))
		pkl.dump(phone_count, open(path_name+"phone_count.pkl",'wb'))
		pkl.dump(loc_count, open(path_name+"loc_count.pkl",'wb'))
		pkl.dump(phone_entropy, open(path_name+"phone_entropy.pkl",'wb'))
		pkl.dump(loc_radii, open(path_name+"loc_radii.pkl",'wb'))
		pkl.dump(num_names, open(path_name+"num_names.pkl",'wb'))
		pkl.dump(num_valid_urls, open(path_name+"num_valid_urls.pkl",'wb'))
		pkl.dump(num_invalid_urls, open(path_name+"num_invalid_urls.pkl",'wb'))
		pkl.dump(num_ads_per_week, open(path_name+"num_ads_per_week.pkl",'wb'))
		pkl.dump(num_urls, open(path_name+"num_urls.pkl",'wb'))
		pkl.dump(num_social, open(path_name+"num_social.pkl",'wb'))
		pkl.dump(num_emails, open(path_name+"num_emails.pkl",'wb'))
	else:
		# Read in cluster features
		cluster_sizes = pkl.load(open(path_name+"cluster_sizes.pkl",'rb'))
		phone_count = pkl.load(open(path_name+"phone_count.pkl",'rb'))
		loc_count = pkl.load(open(path_name+"loc_count.pkl",'rb'))
		phone_entropy = pkl.load(open(path_name+"phone_entropy.pkl",'rb'))
		valid_phone_ratio = pkl.load(open(path_name+"valid_phone_ratio.pkl",'rb'))
		loc_radii = pkl.load(open(path_name+"loc_radii.pkl",'rb'))
		num_names = pkl.load(open(path_name+"num_names.pkl",'rb'))
		num_valid_urls = pkl.load(open(path_name+"num_valid_urls.pkl",'rb'))
		num_invalid_urls = pkl.load(open(path_name+"num_invalid_urls.pkl",'rb'))
		num_ads_per_week = pkl.load(open(path_name+"num_ads_per_week.pkl",'rb'))
		num_urls = pkl.load(open(path_name+"num_urls.pkl",'rb'))
		num_social = pkl.load(open(path_name+"num_social.pkl",'rb'))
		num_emails = pkl.load(open(path_name+"num_emails.pkl",'rb'))
		num_names = pkl.load(open(path_name+"num_names.pkl",'rb'))
		

	# making pair plots of the features
	if not recompute:
		plot_df = pd.read_csv(path_name+"/plot_df.csv", index_col=False)
	else:
		# data needs to be in dataframe format
		plot_df = pd.DataFrame(columns = ["Cluster Size", "Phone Count", "Loc Count", "Phone Entropy", \
							"Loc Radius","Person Name Count",\
							"Valid URLs", "Invalid URLs", "Ads/week", 'Num URLs', "Num Social", \
							"Num Emails", "cluster_id"])

		plot_df['Cluster Size'] = [np.log(clsize) for id, clsize in cluster_sizes.items()]
		plot_df['Phone Count'] = [np.log(clsize) if clsize != 0 else 0 for id,clsize in phone_count.items()]
		plot_df['Loc Count'] = [np.log(clsize) if clsize != 0 else 0 for id,clsize in loc_count.items()]
		plot_df['Phone Entropy'] = [clsize if clsize != 0 else 0 for id,clsize in phone_entropy.items()]
		plot_df['Loc Radius'] = [np.log(clsize) if clsize !=0 else 0 for id,clsize in loc_radii.items()]
		plot_df['Person Name Count'] = [np.log(clsize) if clsize !=0 else 0 for id,clsize in num_names.items()]
		plot_df['Valid URLs'] = [np.log(clsize) if clsize != 0 else 0 for id,clsize in num_valid_urls.items()]
		plot_df['Invalid URLs'] = [np.log(clsize) if clsize != 0 else 0 for id,clsize in num_invalid_urls.items()]
		plot_df['Ads/week'] = [np.log(clsize) if clsize != 0 else 0 for id,clsize in num_ads_per_week.items()]
		plot_df['Num URLs'] = [np.log(clsize) if clsize != 0 else 0 for id,clsize in num_urls.items()]
		plot_df['Num Emails'] = [np.log(clsize) if clsize != 0 else 0 for id,clsize in num_emails.items()]
		plot_df['Num Social'] = [np.log(clsize) if clsize != 0 else 0 for id,clsize in num_social.items()]	

		plot_df['Num Social Val'] = [clsize for id,clsize in num_social.items()]
		plot_df['Num Emails Val'] = [clsize for id,clsize in num_emails.items()]
		plot_df['Cluster Size Val'] = [clsize for id, clsize in cluster_sizes.items()]
		plot_df['Phone Count Val'] = [clsize for id,clsize in phone_count.items()]
		plot_df['Loc Count Val'] = [clsize for id,clsize in loc_count.items()]
		plot_df['Phone Entropy Val'] = [clsize for id,clsize in phone_entropy.items()]
		plot_df['Loc Radius Val'] = [clsize for id,clsize in loc_radii.items()]
		plot_df['Person Name Count Val'] = [clsize for id,clsize in num_names.items()]
		plot_df['Valid URLs Val'] = [clsize for id,clsize in num_valid_urls.items()]
		plot_df['Invalid URLs Val'] = [clsize for id,clsize in num_invalid_urls.items()]
		plot_df['Ads/week Val'] = [clsize for id,clsize in num_ads_per_week.items()]
		plot_df['Num URLs Val'] = [clsize for id,clsize in num_urls.items()]
		plot_df['cluster_id'] = [k for k in cluster_sizes.keys()]
		
		plot_df.to_csv(path_name+"/plot_df.csv",index=False)