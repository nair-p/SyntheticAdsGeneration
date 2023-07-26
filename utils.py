import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from scipy.stats import levy, levy_stable, pareto
import random
import matplotlib.pyplot as plt

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


	for id, row in df.iterrows():
		ad = row.cleaned_text
		duplicate = np.random.rand() > 0.5
		location = row.location
		post_date = row.post_date
		if duplicate: # 60% of the time, duplicate the ad
			keep_same_location = np.random.rand() > 0.5
			if keep_same_location:
				size = cluster_sizes[id]
				full_ads.extend([ad] * size)
				locations.extend([location] * size)
				
			else:

		else: # 40% of the time, make small variations



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
		first = makeFirst()
		second = makeSecond()
		last = makeLast()
		return '{3}-{3}-{4}'.format(first,second,last)

	numbers = []
	for _ in tqdm(range(size)):
		numbers.append(makePhone)

	return numbers

def load_data(filename):
	# this function loads the csv file from the given filename
	'''
	input: filename - path to the csv file containing the unique starter ads
	output: data - csv file loaded from the given path as a csv file
	'''
	data = pd.read_csv(filename, index_col=False,nrows=100)
	if 'cleaned_text' in data.columns:
		data.rename(columns={'cleaned_text':'description'},inplace=True)
	return data