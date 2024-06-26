from tqdm import tqdm
import numpy as np
import pandas as pd
import ast
import random
import networkx as nx
from utils import generate_random_phones, random_date
from itertools import combinations
from igraph import *
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
from collections import Counter

def add_edge_info(df):
	# function to explicitly add edges/connections among clusters
	'''
	input: df - dataframe with clusters that need to be conencted using meta-data
	output: df - dataframe with connections
	'''
	# connections can be by means of 3 meta-data sources
	'''
	1. phone numbers
	2. social media tags
	3. email ids
	'''

	def add_links(data, connections):
		for connection in tqdm(connections):
			e1 = connection[0]
			e2 = connection[1]

			df1 = data[data.cluster_id==e1]
			df2 = data[data.cluster_id==e2]

			meta_choice = np.random.choice(['phone','social','email'])

			val1 = df1[meta_choice].values
			val2 = df2[meta_choice].values


			for idx, val in zip(df1.index.values, val2[:len(df1)]):
				data.loc[idx, meta_choice] = val

		return data

	clusters = df.cluster_id.unique()
	number_of_clusters = len(clusters)
	number_of_meta_clusters = 720
	mean_cluster_size = number_of_clusters / number_of_meta_clusters
	shape, mode = 1., 10  # shape and mode of the pareto distribution
	lower = mean_cluster_size/2
	upper = 2*mean_cluster_size

	clus_ind_map = dict(zip(df['cluster_id'].unique(),range(number_of_clusters)))
	micro_to_meta_map = np.zeros([number_of_clusters, number_of_clusters])
	meta_id = 0
	available_clusters = set(clusters)
	meta_label_map = {}

	for meta_cluster in tqdm(range(number_of_meta_clusters)):
	# while tqdm(len(available_clusters) > 0):
		size = np.random.pareto(shape, 1) + lower
		size = int(size[0])
		while size > upper:
			size = np.random.pareto(shape, 1) + lower
			size = int(size[0])

		nodes_in_comp = np.random.choice(list(available_clusters), size=size, replace=False)
		for x in nodes_in_comp:
			meta_label_map[x] = meta_id

		meta_id += 1
		
		indices = list(combinations(nodes_in_comp, 2))
		for index in indices:
			micro_to_meta_map[clus_ind_map[index[0]]][clus_ind_map[index[1]]] += 1

		# randomly sample number of connections
		num_phone_connections = np.random.choice(range(len(indices))) # arbitrarily chosen values
		num_social_connections = np.random.choice(range(len(indices)))
		num_email_connections = np.random.choice(range(len(indices)))

		# df = add_links(df, indices)

		available_clusters = available_clusters - set(nodes_in_comp)

	# remaining nodes
	for x in available_clusters:
		meta_label_map[x] = meta_id
	

	df['Meta label'] = df['cluster_id'].apply(lambda x:meta_label_map[x])
	
	all_cluster_sizes = []
	for cl, grp in df.groupby("Meta label"):
		all_cluster_sizes.append(grp.cluster_id.nunique())

	print(Counter(all_cluster_sizes))

	return df


def add_email_address(df):
	# function to create email ids based off of the names. 
	'''
	input: df - dataframe containing starter ads and names advertised
	output: df - dataframe with a "email" column for the added email ids
	'''
	print("Adding email addresses...")
	number_of_ads = len(df)
	email_tags = [
	"@mail.com",
	"@email.com",
	"@look.com"
	]

	email_prefix = [
	"mailme_",
	"",
	"msg.",
	"txt."
	]

	emails = []
	for id, row in df.iterrows():
		names = row.names
		if names == '[]':
			emails.append(np.nan)
		else:
			names_list = ast.literal_eval(names)
			selected_name = np.random.choice(names_list, size=1)[0]
			selected_tag = np.random.choice(email_tags, size=1)[0]
			selected_prefix = np.random.choice(email_prefix, size=1)[0]
			emails.append(selected_prefix+selected_name.lower()+selected_tag)

	df['email'] = emails
	return df


def add_social_media(df):
	# function to create social media tags based off of the names. 
	'''
	input: df - dataframe containing starter ads and names advertised
	output: df - dataframe with a "social" column for the added SM tags
	'''
	print("Adding social media....")
	number_of_ads = len(df)
	social_tags = [
	"_xoxo",
	"_xxx",
	"xx"
	"123",
	"_89"
	]
	socials = []
	for id, row in df.iterrows():
		names = row.names
		if names == '[]':
			socials.append(np.random.choice(['reachme@', np.nan],size=1)[0])
		else:
			names_list = ast.literal_eval(names)
			selected_name = np.random.choice(names_list, size=1)[0]
			selected_suffix = np.random.choice(social_tags, size=1)[0]
			socials.append(selected_name.lower() + selected_suffix)

	df['social'] = socials
	return df

def add_phone_numbers(df, random_generation):
	# function to add randomly sampled phone numbers from a given list or randomly generated if not
	'''
	input: df - dataframe of ads
		   random_generation - boolean variable. If true, numbers are randomly generated. Else, they are sampled from a txt file and mapped to md5 hash.
	return: df - dataframe with phone number and hashed phone numbers
	'''
	print("Generating phone numbers...")
	number_of_ads = len(df)
	if random_generation:
		phone_numbers = generate_random_phones(size=number_of_ads)
	else:
		phones = open("phone_numbers.txt",'r').read().split("\n")
		phone_numbers = np.random.choice(phones, size=number_of_ads)
		
	import hashlib
	hashed_numbers = []
	for number in phone_numbers:
		hashed_numbers.append(hashlib.md5(number.encode('utf-8')).hexdigest())

	df['phone'] = phone_numbers
	df['phone_hashed'] = hashed_numbers

	return df

def add_locations(df):
	# function to add randomly selected locations to the dataframe. We use a list of locations read from a txt file
	'''
	input: df - dataframe of starter ads
	output: df - dataframe containing locations
	'''
	locations_list = pd.read_csv("locations.csv",index_col=False).city.values
	number_of_ads = len(df)

	print("Adding locations...")
	locations = np.random.choice(locations_list, size=number_of_ads)
	df['location'] = locations
	return df


def add_post_date(df):
	# function to add a random post date. All posting dates are between Jan 1, 2022 to June 1, 2022
	'''
	input: df - dataframe of starter ads
	output: df - dataframe with randomly generated post-dates
	'''
	    
	number_of_ads = len(df)
	post_dates = []

	print("Generating dates between 1/1/2022 and 1/6/2022...")
	for _ in tqdm(range(number_of_ads)):
		post_dates.append(random_date("1/1/2022", "6/1/2022", random.random()))

	df['post_date'] = post_dates
	df.sort_values(by='post_date',inplace=True)
	return df

def add_meta_data(df, random_generation=False):
	# this function adds the relevant meta-data information as columns to the input data frame
	'''
	input:  df - dataframe containing starter ads

	output: df - dataframe with additional columns for post-date, location and other contact info
	'''

	df = add_post_date(df)
	df = add_locations(df)
	df = add_phone_numbers(df, random_generation)
	df = add_social_media(df)
	df = add_email_address(df)

	return df
