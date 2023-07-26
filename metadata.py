from tqdm import tqdm
import numpy as np
import pandas as pd
import ast
from utils import generate_random_phones

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
	import random
	import time
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

	    return time.strftime(time_format, time.localtime(ptime))


	def random_date(start, end, prop):
	    return str_time_prop(start, end, '%m/%d/%Y', prop)
	    
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

	return df