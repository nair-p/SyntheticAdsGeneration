import argparse

from metadata import *
from utils import *
from activity_insertion import *


def parse_args():
	parser = argparse.ArgumentParser(description='Synthetic data generation')
	parser.add_argument('--starter_file', type=str, 
                    help='path to the csv file containing starter ads')
	parser.add_argument('--random_phones', action='store_true', 
					help='Indicate if phone numbers should be randomly generated. If false, they will be sampled from an input file in the same path.')
	parser.add_argument("--only_mo", action='store_true',
					help='Indicate whether dataset with clusters already exists and only activity insertion needs to be done. \
					If True, clusters wont be added.')
	parser.add_argument("--only_edges", action='store_true',default=False,
					help='Indicate whether to add connections between clusters. If false, the code assumes a data file with links already exists.')
	args = parser.parse_args()
	return args

def create_dataset_with_micro_clusters(args):
	# load data
	starter_data = load_data(args.starter_file)

	# adding all relevant meta data information
	df_with_meta_data = add_meta_data(starter_data, args.random_phones)
	# save starter df with metadata
	df_with_meta_data.to_csv(args.starter_file.split(".csv")[0]+"_with_metadata.csv",index=False)

	# adding duplicates and inserting micro-clusters
	df_with_micro_clusters = add_clusters(df_with_meta_data)

	return df_with_micro_clusters

def main():
	# get arguments
	args = parse_args()
	
	if not args.only_mo:
		# adding micro-clusters
		df_with_micro_clusters = create_dataset_with_micro_clusters(args)
		save_file_name = args.starter_file.split(".csv")[0] + "_with_clusters.csv"

		print("Size of augmented dataset = " + str(df_with_micro_clusters.shape))
		df_with_micro_clusters.to_csv(save_file_name, index=False)

	else:
		file_name = args.starter_file.split(".csv")[0] + "_with_clusters.csv"
		df_with_micro_clusters = pd.read_csv(file_name, index_col=False)

	save_file_name = args.starter_file.split(".csv")[0] + "_with_edges.csv"
	if args.only_edges: # add edges
		df_with_connections = fill_meta_data(df_with_micro_clusters)
		df_with_connections.to_csv(save_file_name, index=False)

	else:
		df_with_connections = pd.read_csv(save_file_name, index_col=False)


	save_file_name = args.starter_file.split(".csv")[0] + "_with_mo_spam.csv"
	df_with_connections['mo_labels'] = None

	# adding M.Os
	print("Adding spam...")
	df_with_mo = add_activity(df_with_connections, mo_type='spam')
	df_with_mo.to_csv(save_file_name, index=False)
	
	print("Adding HT...")
	save_file_name = args.starter_file.split(".csv")[0] + "_with_mo_ht.csv"
	df_with_mo = add_activity(df_with_mo, mo_type='ht')
	df_with_mo.to_csv(save_file_name, index=False)
	# df_with_mo = pd.read_csv(save_file_name, index_col=False)
	
	print("Adding ISW...")
	save_file_name = args.starter_file.split(".csv")[0] + "_with_mo_isw.csv"
	df_with_mo = add_activity(df_with_mo, mo_type='isw')
	print("Size of augmented dataset = " + str(df_with_mo.shape))
	df_with_mo.to_csv(save_file_name, index=False)
	
if __name__ == '__main__':
	main()