from utils import *
import pandas as pd
from tqdm import tqdm
import random

def add_ht(df, ht_percent=0.2):
	# this function adds HT activity to the df
	'''
	HT LFS:
		1. 1-5 instances of HT related keywords
		2. Number of names > 3
		3. Movement indication (number of locations > 2 within 5 days time span)

	input: df - dataframe with clustered ads
	output: df - dataframe with added HT
	'''
	THIRD_PERSON = ["she", "her", "hers", "herself"] # Third person pronouns.
	FIRST_PERSON_PLURAL = ["us", "our", "ours", "ourselves"] # First person plural pronouns

	PRONOUNS = THIRD_PERSON + FIRST_PERSON_PLURAL

	NON_RESTRICTED_SEX_SEQUENCE = ["bb", "raw", "bareback", "bare back", "no cover", "without condom", "no condom"]
	NON_RESTRICTED_SEX_SEQUENCE += ['BBBJ',"BBBj",'bxxJ','b88j','B-B-B-J']

	INCALL_ONLY = ["in call", "in call only", "in-call only", "incall only"]
	NO_OUTCALL = ["no outcall", "no out call", "no out-call"]

	HT_KEYWORDS = ["new in town", "new girl", "came to the town", "out off town", 
				   "few days",  "back to town", "out of town", "in your town", 
				   "for a few days", "back in town",  "only stay for a few", "new arrived", 
				   "just arrived", "new here", "new to this",  "new agency", "new in town", 
				   "new to your city", "i'm new", "i am new", "brand new",  "little new",
				   "very new", "look new", "short-term", "new to the area", "new to the industry", 
				   "new in the business", "new face", "short time", "in town for", "new to town", 
				   "short stay", "short term", "short time", "short period", "arrived in town", 
				   "arrived short term", "for a limited time", "newly arrived", "passing through", 
				   "on vacation", "in town for a few days", "i am a newb", "i am a newbie", "first time to", 
				   "first time in", "first time here", "first time visiting", "new in the field",
				   "just turned 18", "turned 18", "hot teen", "fresh", "petite", "student", "college", "young",
				   "juicy", "tight"]


	number_of_clusters = df.cluster_id.nunique()
	number_of_ht_clusters = int(ht_percent*number_of_clusters)
	available_clusters = df[pd.isna(df.mo_labels)].cluster_id.unique
	ht_cluster_ids = np.random.choice(available_clusters, size=number_of_ht_clusters, replace=False)
	ht_idx = df[df.cluster_id.isin(ht_cluster_ids)].index.values
	mo_labels = df.mo_labels.values
	mo_labels[ht_idx] = 'ht'

	df['mo_labels'] = mo_labels
	


def add_spam(df, spam_percent=0.3):
	# this function adds spam like activity to the df
	'''
	SPAM LFS:
		1. Number of locations > 10
		2. Number of phones > 5
		3. Location radius over time >> ( > 100 arbitrarily chosen)

	input: df - dataframe with clustered ads
	output: df - dataframe with added spam
	'''
	number_of_clusters = df.cluster_id.nunique()
	number_of_spam_clusters = int(spam_percent*number_of_clusters)
	available_clusters = df[pd.isna(df.mo_labels)].cluster_id.unique
	spam_cluster_ids = np.random.choice(available_clusters, size=number_of_spam_clusters, replace=False)
	spam_idx = df[df.cluster_id.isin(spam_cluster_ids)].index.values
	mo_labels = df.mo_labels.values
	mo_labels[spam_idx] = 'spam'

	df['mo_labels'] = mo_labels
	all_dates = df.post_dates.values

	for cluster_id in spam_cluster_ids:
		grp = df[df.cluster_id==cluster_id]
		number_of_locations = grp.location.nunique()
		if number_of_locations < 10:
			locations_list = pd.read_csv("locations.csv",index_col=False).city.values
			new_locations = np.random.choice(locations_list, size=10-number_of_locations)
			idx = np.random.choice(grp.index.values, size=10-number_of_locations, replace=False)
			for id, loc in zip(idx, new_locations):
				df.loc[id, 'location'] = loc

		number_of_phones = grp.phone.nunique()
		if number_of_phones < 5: # adding more phone numbers
			phones = grp.phone.unique()
			num_phones = np.random.choice(range(5,10),size=1)[0]
			new_phones = generate_random_phones(num_phones)
			idx = np.random.choice(grp.index.values, size=num_phones, replace=False)
			for id, ph in zip(idx, new_phones):
				df.loc[id, 'phone'] = ph

		posting_dates = pd.to_datetime(grp.post_dates, infer_datetime_format=True)
		dates = sorted(posting_dates)
		diff = (dates[-1]-dates[0])/np.timedelta64(1,'D')
		
		# add bursty behavior
		number_of_bursts = max(1, int(diff/4))
		bursty_dates = np.random.choice(posting_dates, size=number_of_bursts, replace=False)
		repeats = int(np.ceil(len(grp)/number_of_bursts)) # how many times should the bursty dates be repeated

		modified_dates = []
		for date in bursty_dates:
			modified_dates.extend([date]*repeats)

		random.shuffle(modified_dates)
		all_dates[grp.index.values] = modified_dates[:len(grp)]

	df['post_dates'] = pd.to_datetime(all_dates)
	df['post_dates'] = df.post_dates.dt.strftime('%Y-%m-%d')

	return df


def add_activity(df, mo_type):
	# this function adds activity of a particular type into the passed dataframe
	'''
	input: df - dataframe containing clustered ads and metadata
		   mo_type - the type of activity to insert. Eg 'spam', 'ht' or 'isw'

	output: df - dataframe with the mo_type ads inserted
	'''
	df['mo_labels'] = None
	if mo_type == 'spam':
		df = add_spam(df, spam_percent=0.3)
	elif mo_type == 'ht':
		df = add_ht(df, ht_percent=0.2)

	return df