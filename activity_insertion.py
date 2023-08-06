from utils import *
import pandas as pd
from tqdm import tqdm
import random
import ast
from datetime import timedelta

def add_isw(df, isw_percent=0.5):
	# this function adds ISW activity to the df
	'''
	ISW LFS:
		1. 1-5 instances of ISW related keywords
		2. Number of names in [1, 2] with [1, 2] phone numbers

	input: df - dataframe with clustered ads
	output: df - dataframe with added ISW
	'''
	## ISW Indicators
	WITH_CONDOM_SEQUENCE = ["with condom", "use of condoms", "with a condom", "no bb", "safe", "safe play", "covered"]
	RESTRICTED_SEX_SEQUENCE = ["no oral", "no anal", "no black", "no greek", "gentlemen only", "respectful"]
	RESTRICTED_SEX_SEQUENCE += WITH_CONDOM_SEQUENCE

	INCALL_WORDS = ["incall", "in-call", "incalls", 'in call']
	OUTCALL_WORDS = ["outcall", "out-call", "outcalls", 'out call']
	CARCALL_WORDS = ["carcall", "car-call", "cardate", 'carplay']
	NO_INCALL_SEQUENCE = ["no incalls", "no incall", "no in-calls", "no in calls", "no incall"]
	OUTCALL_SEQUENCE = ["out call", "out call only", "out-call only", "outcall only"]

	SW_INCALL_WORDS = INCALL_WORDS + OUTCALL_SEQUENCE + CARCALL_WORDS + NO_INCALL_SEQUENCE + OUTCALL_WORDS

	SW_KEYWORDS = ["mature", "classy", "real", "deposit", 'advance', "professional", "appointment", 'milf', 'mommy']

	number_of_clusters = df.cluster_id.nunique()
	number_of_isw_clusters = int(isw_percent*number_of_clusters)
	available_clusters = df[df.mo_labels.isna()].cluster_id.unique()
	isw_cluster_ids = np.random.choice(available_clusters, size=number_of_isw_clusters, replace=False)
	isw_idx = df[df.cluster_id.isin(isw_cluster_ids)].index.values
	mo_labels = df.mo_labels.values
	mo_labels[isw_idx] = 'isw'

	df['mo_labels'] = mo_labels
	list_of_names = set(beautify(df.names.values))

	clusters_with_2_phones = np.random.choice(isw_cluster_ids, size=int(0.05*number_of_isw_clusters))
	clusters_with_1_phone = list(set(isw_cluster_ids) - set(clusters_with_2_phones))

	for cluster_id in tqdm(isw_cluster_ids):
		grp = df[df.cluster_id==cluster_id]
		number_of_locations = grp.location.nunique()
		grp_names = beautify(grp.names.values)
		if len(grp_names) == 0:
			grp_names = beautify(df.names.values)
		number_of_names = len(np.unique(grp_names))
		grp_phones = grp.phone.unique()

		# adding ISW related keywords
		number_of_instances_to_add = np.random.choice(range(1,6), size=1)[0]
		type_of_kw_to_add = np.random.choice(['restrictions', 'isw_keywords', 'calls'],\
			size=number_of_instances_to_add)

		words_to_add = []
		for type_to_add in type_of_kw_to_add:
			if type_to_add == 'restrictions':
				words_to_add.append(np.random.choice(RESTRICTED_SEX_SEQUENCE))
			elif type_to_add == 'isw_keywords':
				words_to_add.append(np.random.choice(SW_KEYWORDS))
			else:
				words_to_add.append(np.random.choice(SW_INCALL_WORDS))
		for ad, ind in zip(grp.cleaned_text.values, grp.index.values):
			words = ad.split(" ")
			
			idx_to_insert = np.random.choice(range(len(words)), size=len(words_to_add))
			start_idx = idx_to_insert[0]
			new_ad = " ".join(x for x in words[:start_idx])

			for id, idx in enumerate(idx_to_insert[1:]):
				new_ad += " ".join(x for x in words[start_idx:idx])
				new_ad += (" "+words_to_add[id]+" ")
				start_idx = idx

			new_ad += " ".join(x for x in words[start_idx:])
			df.loc[ind, 'cleaned_text'] = new_ad

		if cluster_id in clusters_with_1_phone:
			phone = np.random.choice(grp_phones)
			phones = [phone]*len(grp)
			name = np.random.choice(grp_names)
			names = [name] * len(grp)
		else:
			if len(grp_phones) >= 2:
				phone = np.random.choice(grp_phones, size=2)
			else:
				phone = [grp_phones[0], grp_phones[0]]
			phones = [phone[0]]*int(np.ceil(len(grp)/2))
			phones.extend([phone[1]]*int(np.ceil(len(grp)/2)))

			name = np.random.choice(grp_names, size=2)
			names = [name[0]] * int(np.ceil(len(grp)/2))
			names.extend([name[1]]*int(np.ceil(len(grp)/2)))

		for i, indx in enumerate(grp.index.values):
			df.loc[indx, 'phone'] = phones[i]
			df.loc[indx, 'names'] = names[i]

	return df

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
	CALLS = INCALL_ONLY + NO_OUTCALL

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

## patterns are similar in feature matrix (compare degree distribution, etc)
# same type of graphs instead of same graphs
# diff between degree distributions, number of connected components, size of connected components
# get french keywords from Sophia

	number_of_clusters = df.cluster_id.nunique()
	number_of_ht_clusters = int(ht_percent*number_of_clusters)
	available_clusters = df[df.mo_labels.isna()].cluster_id.unique()
	ht_cluster_ids = np.random.choice(available_clusters, size=number_of_ht_clusters, replace=False)
	ht_idx = df[df.cluster_id.isin(ht_cluster_ids)].index.values
	mo_labels = df.mo_labels.values
	mo_labels[ht_idx] = 'ht'

	df['mo_labels'] = mo_labels
	list_of_names = set(beautify(df.names.values))
	df['post_dates'] = pd.to_datetime(df.post_dates)
	df['post_dates'] = df.post_dates.dt.strftime('%Y-%m-%d')

	for cluster_id in ht_cluster_ids:
		grp = df[df.cluster_id==cluster_id]
		number_of_locations = grp.location.nunique()
		grp_names = beautify(grp.names.values)
		number_of_names = len(np.unique(grp_names))

		# adding HT related keywords
		number_of_instances_to_add = np.random.choice(range(1,6), size=1)[0]
		type_of_kw_to_add = np.random.choice(['pronouns', 'restrictions', 'ht_keywords', 'calls'],\
			size=number_of_instances_to_add)

		words_to_add = []
		for type_to_add in type_of_kw_to_add:
			if type_to_add == 'pronouns':
				words_to_add.append(np.random.choice(PRONOUNS))
			elif type_to_add == 'restrictions':
				words_to_add.append(np.random.choice(NON_RESTRICTED_SEX_SEQUENCE))
			elif type_to_add == 'ht_keywords':
				words_to_add.append(np.random.choice(HT_KEYWORDS))
			else:
				words_to_add.append(np.random.choice(CALLS))
		for ad, ind in tqdm(zip(grp.cleaned_text.values, grp.index.values)):
			words = ad.split(" ")
			
			idx_to_insert = np.random.choice(range(len(words)), size=len(words_to_add))
			start_idx = idx_to_insert[0]
			new_ad = " ".join(x for x in words[:start_idx])

			for id, idx in enumerate(idx_to_insert[1:]):
				new_ad += " ".join(x for x in words[start_idx:idx])
				new_ad += (" "+words_to_add[id]+" ")
				start_idx = idx

			new_ad += " ".join(x for x in words[start_idx:])
			df.loc[ind, 'cleaned_text'] = new_ad

		# ensuring more than 2 people are advertised
		if number_of_names < 3:
			remaining_names = list_of_names - set(grp_names)
			name_count = np.random.choice(range(3,6))
			new_names = np.random.choice(list(remaining_names), size=name_count-3, replace=False)
			idx_to_add_names = np.random.choice(grp.index.values, size=len(new_names))
			for id, indx in tqdm(enumerate(idx_to_add_names)):
				names = grp.names.values[id]
				if len(names) == 0:
					names = []
				elif names[0] == '[':
					names = ast.literal_eval(names)
				else:
					names = [names]
				names.append(new_names[id])
				df.loc[indx, 'names'] = str(names)

		# movement indication
		if number_of_locations > 1:
			posting_dates = pd.to_datetime(grp.post_dates, infer_datetime_format=True)
			location = grp.location.unique()[0] # base location
			new_locations = get_locations_within_radius(location, size=len(grp), radius=200) # provide radius in kms
			# new_locations = [locations[0]]*int(np.ceil(len(grp)/2))
			# new_locations.extend([locations[1]]*int(np.ceil(len(grp)/2)))
			
			dates = sorted(posting_dates)
			diff = (dates[-1]-dates[0])/np.timedelta64(1,'D')
			new_dates = []
			if diff < 4: # if movement happened in an unreasonable amount of time, then adjust it
				for _ in tqdm(range(len(grp))):
					start = dates[0].strftime('%Y-%m-%d')
					end = dates[0]+timedelta(days=5)
					end = end.strftime('%Y-%m-%d')
					date = random_date(start, end, random.random())
					if pd.isnull(date):
						date = random_date(start, end, random.random())
					new_dates.append(date)
			else:
				new_dates = posting_dates.values
			i = 0
			if len(new_locations) < len(grp):
				last_loc = new_locations[-1]
				for _ in range(len(grp)-len(new_locations)):
					new_locations.append(last_loc)
			for ind, row in grp.iterrows():
				df.loc[ind, 'location'] = new_locations[i]
				df.loc[ind, 'post_dates'] = new_dates[i]
				i+=1

	df['post_dates'] = pd.to_datetime(df.post_dates)
	df['post_dates'] = df.post_dates.dt.strftime('%Y-%m-%d')

	return df


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
	available_clusters = df[df.mo_labels.isna()].cluster_id.unique()
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
	
	if mo_type == 'spam':
		df = add_spam(df, spam_percent=0.3)
	elif mo_type == 'ht':
		df = add_ht(df, ht_percent=0.2)
	elif mo_type == 'isw':
		df = add_isw(df, isw_percent=0.5)

	return df