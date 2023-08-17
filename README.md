# SyntheticAdsGeneration
Code for adding M.Os to synthetically generated ads using GPT3.5

This code-base assumes that the starter files (in csv format) contains a corpus of synthetically generated ads and follows the below-mentioned pipeline for adding specific kinds of activities. 

The pipeline followed for the synthetic generation is given below.
![synthetic_data_generation_workflow.png](https://github.com/nair-p/SyntheticAdsGeneration/blob/main/figs/synthetic_data_generation_workflow.png)

First, we want to add posting date, location, social media tags and phone numbers.
For extracting names, we use [HTNER](www.github.com/HTNER). 



# double check if the graph info is correct
node and its neighbors have the same label
check if meta-clusters have the same activity
after getting micro-clusters, get the meta-clusters and then M.O injection should be at the meta-cluster level (all micro-clusters in that meta-cluster). All should have the same label.