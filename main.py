import argparse

from metadata import *
from utils import *


def parse_args():
	parser = argparse.ArgumentParser(description='Synthetic data generation')
	parser.add_argument('--starter_file', type=str, 
                    help='path to the csv file containing starter ads')
	parser.add_argument('--random_phones', action='store_true', 
					help='Indicate if phone numbers should be randomly generated. If false, they will be sampled from an input file in the same path.')
	args = parser.parse_args()
	return args

def create_dataset_with_micro_clusters(args):
	# load data
	starter_data = load_data(args.starter_file)

	# adding all relevant meta data information
	df_with_meta_data = add_meta_data(starter_data, args.random_phones)

	# adding duplicates and inserting micro-clusters
	df_with_micro_clusters = add_clusters(df_with_meta_data)

	return df_with_micro_clusters

def main():
	# get arguments
	args = parse_args()

	# adding micro-clusters
	df_with_micro_clusters = create_dataset_with_micro_clusters(args)
	save_file_name = args.starter_file.split(".csv")[0] + "_with_clusters.csv"

	print("Size of augmented dataset = " + str(df_with_micro_clusters.shape))
	df_with_micro_clusters.to_csv(save_file_name, index=False)


if __name__ == '__main__':
	main()