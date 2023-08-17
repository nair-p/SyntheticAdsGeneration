import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from scipy.stats import levy, levy_stable, pareto
import random
import matplotlib.pyplot as plt
import datetime
import ast
from tqdm import tqdm
import haversine as hs
import time
import geopandas as gpd
import shapely
from itertools import combinations
import networkx as nx

def get_locations_within_radius(city, size, radius):
	# this function takes a city name and radius as input and returns locations within that radius
	'''
	input: city - name of the base location
		   size - number of cities required within the radius
		   radius - maximum radius of distance to be considered

	output: nearby_points - city names within the given radius of the given city
	'''
	import libpysal
	from libpysal.cg.kdtree import KDTree 

	# data file containing the lat, lon info of cities in the country
	location_info = pd.read_csv("locations.csv",index_col=False)
	city_info = location_info[location_info.city==city]

	locations = list(zip(location_info.xcoord, location_info.ycoord))
	tree = KDTree(locations, distance_metric='Arc', radius=libpysal.cg.RADIUS_EARTH_MILES)
	current_point = (city_info.xcoord.tolist()[0], city_info.ycoord.tolist()[0])

	# get all points within 1 mile of 'current_point'
	indices = tree.query_ball_point(current_point, radius)
	potential_points = np.array(locations)[indices]

	potential_cities = []
	for point in potential_points:
		potential_cities.extend(location_info[(location_info.xcoord==point[0]) & (location_info.ycoord==point[1])].city.values)

	return potential_cities[:size]



def beautify(given_list):
	# this function takes a list of list with non-constant size and ravels it into a single list
	'''
	input: given_list - list of lists
	output: ravelled_list - single ravelled list of input
	'''
	new_list = []
	for item in given_list:
		if type(item) != list:
			if item[0] == '[':
				new_list.append(ast.literal_eval(item))
			else:
				new_list.append([item])
		else:
			new_list.append(item)

	# given_list = [ast.literal_eval(x) if x!= "" and x[0] == '[' else [x] for x in given_list]
	ravelled_list = []
	for item in new_list:
		ravelled_list.extend(item)
	return np.unique(ravelled_list)

def find_loc_radii(list_locs):
	# this function finds the maximum radius covered by a given list of locations
	'''
	input: list_locs - list of locations as (lat, lon) pairs
	output: radius - maximum radius covered by the farthest locations provided in the list
	'''
	locations_info = pd.read_csv("locations.csv", index_col=False)
	loc_geo = locations_info[locations_info['city'].isin(list_locs)][['xcoord','ycoord']].values

	t_lat = sorted(loc_geo, key=lambda x: float(x[0]),reverse=True)
	t_lon = sorted(loc_geo, key=lambda x: float(x[1]),reverse=True)

	x11 = float(t_lat[0][0])
	y11 = float(t_lat[0][1])
	x21 = float(t_lat[-1][0])
	y21 = float(t_lat[-1][1])
	
	x12 = float(t_lon[0][0])
	y12 = float(t_lon[0][1])
	x22 = float(t_lon[-1][0])
	y22 = float(t_lon[-1][1])
	
	d1 = hs.haversine((x11,y11),(x21,y21))
	d2 = hs.haversine((x12,y12),(x22,y22))
	radius = max(d1, d2)
	return radius

def get_list_of_names(df):
	names = df.names
	list_of_names = []
	for name in names:
		name_list = ast.literal_eval(name)
		list_of_names.extend(name_list)

	list_of_names = list(set(list_of_names))
	return list_of_names


# below code is from https://stackoverflow.com/questions/553303/generate-a-random-date-between-two-other-dates
def str_time_prop(start, end, time_format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formatted in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, time_format))
    etime = time.mktime(time.strptime(end, time_format))

    ptime = stime + prop * (etime - stime)
    # print(stime, prop, etime, stime)

    return time.strftime(time_format, time.localtime(ptime))


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

def random_date(start, end, prop):
    return str_time_prop(start, end, '%Y-%m-%d', prop)

def add_clusters(df, total_size=1000000):
	# function to duplicate ads and inject micro-clusters
	'''
	input: df - dataframe with starter ads and associated meta-data
		   total_size - number of ads in the final dataset
	output: new_df - dataframe containing "total_size" ads 
	'''

	# each ad in the input df acts as a cluster since they are all unique
	# we sample the cluster sizes from a pareto distribution
	number_of_clusters = len(df)
	# number_of_clusters = 7000
	mean_cluster_size = total_size / number_of_clusters

	shape, mode = 1., 10  # shape and mode of the pareto distribution
	lower = 100
	cluster_sizes = np.random.pareto(shape, number_of_clusters) + lower

	full_ads = []
	cluster_labels = []
	locations = []
	posting_dates = []
	all_names = []
	all_phones = []
	all_socials = []
	all_emails = []

	names_list = get_list_of_names(df)

	locations_list = pd.read_csv("locations.csv",index_col=False).city.values

	for id, row in df.iterrows():
		ad = row.description
		duplicate = np.random.rand() > 0.5
		location = row.location
		phone = row.phone
		email = row.email
		social = row.social
		post_date = datetime.datetime.strptime(row.post_date, "%m/%d/%Y")
		names = ast.literal_eval(row.names)
		size = int(cluster_sizes[id])

		cluster_labels.extend([id]*size)
		all_phones.extend([phone]*size)
		all_emails.extend([email]*size)
		all_socials.extend([social]*size)
		possible_dates = []
		for day in range(10):
			possible_dates.append(post_date + datetime.timedelta(days=day))
			possible_dates.append(post_date - datetime.timedelta(days=day))

		posting_dates.extend(np.random.choice(possible_dates, size=size))
		if duplicate: # 60% of the time, duplicate the ad
			keep_same_location = np.random.rand() > 0.5
			full_ads.extend([ad] * size)
			all_names.extend([names]*size)

			if keep_same_location:
				locations.extend([location] * size)
			else:
				new_location = np.random.choice(list(set(locations_list)-{location}))
				locations.extend([new_location] * size)

		else: # 40% of the time, make small variations
			number_of_words = len(ad.split())
			percent_words_to_change = np.random.choice(range(6), size=size)
			locations.extend([location] * size)

			change_words = np.random.rand() > 0.5

			if change_words:
				all_names.extend([names]*size)
				for percent in percent_words_to_change:
					number_words_to_change = int(percent/100*number_of_words)
					idx_to_remove = np.random.choice(range(number_of_words), size=number_words_to_change, replace=False)
					words_list = np.array(ad.split())
					words_list[idx_to_remove] = ""
					full_ads.append(" ".join(x for x in words_list))
			else:
				full_ads.extend([ad] * size)
				all_names.extend(np.random.choice(list(set(names_list) - set(names)), size=size))

	new_df = pd.DataFrame(columns=['cleaned_text', 'cluster_id', 'location', 'names', 'post_dates'])
	new_df['cleaned_text'] = full_ads
	new_df['cluster_id'] = cluster_labels
	new_df['location'] = locations
	new_df['names'] = all_names
	new_df['post_dates'] = posting_dates
	new_df['phone'] = all_phones
	new_df['social'] = all_socials
	new_df['email'] = all_emails
	return new_df


def generate_random_phones(size):
	#function to randomly generate valid phone-numbers. This code was from https://stackoverflow.com/a/26226850
	'''
	input: size - number of phone numbers to generate
	output: numbers - list of randomly gennerated phone numbers of size "size"
	'''
	def makeFirst(): # area code with no leading 0s
		first_digit = random.randint(1,9)
		remaining = random.randint(0,99)
		return first_digit*100 + remaining

	def makeSecond(): # middle 3 digits 
		middle1 = random.randint(0,9)
		middle2 = random.randint(0,9)
		middle3 = random.randint(0,9)
		middle = 100*middle1 + 10*middle2 + middle3
		return middle

	def makeLast(): # last 4 digits
		return ''.join(map(str, random.sample(range(10),4)))

	def makePhone(): # putting everything together
		first = str(makeFirst())
		second = str(makeSecond())
		last = str(makeLast())
		return first+"-"+second+"-"+last
		# return '{3}-{3}-{4}'.format(first,second,last)

	numbers = []
	for _ in range(size):
		numbers.append(makePhone())

	return numbers

def load_data(filename):
	# this function loads the csv file from the given filename
	'''
	input: filename - path to the csv file containing the unique starter ads
	output: data - csv file loaded from the given path as a csv file
	'''
	data = pd.read_csv(filename, index_col=False)
	if 'cleaned_text' in data.columns:
		data.rename(columns={'cleaned_text':'description'},inplace=True)
	return data