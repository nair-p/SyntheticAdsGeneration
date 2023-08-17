# SyntheticAdsGeneration
Code for adding M.Os to synthetically generated ads using GPT3.5

This code-base assumes that the starter files (in csv format) contains a corpus of synthetically generated ads and follows the below-mentioned pipeline for adding specific kinds of activities. 

The pipeline followed for the synthetic generation is given below.
![synthetic_data_generation_workflow.png](https://github.com/nair-p/SyntheticAdsGeneration/blob/main/figs/synthetic_data_generation_workflow.png)

Starting with generated ads, we want to add posting date, location, social media tags and phone numbers. For extracting names, we use [HTNER](www.github.com/HTNER). 

Then we manually inject clusters (called micro-clusters) by treating each of the starter ad as a cluster center and duplicating it `n` times where `n` is the size of the cluster sampled from a pareto distribution. 

Once clusters are created, we group them into larger clusters (called meta-clusters) and randomly add links between them by giving them the same phone numbers, social media tags and email ids. All the micro-clusters within a larger meta-cluster will have the same M.O label.  

This can be achieved by running
```
python3 main.py --starter_file \path\to\starter\csv\file\of\generated\ads
```

Once this is done, we need to build the graph by first extracting node features.
```
python3 extract_features.py --filename \path\to\csv\file\with\mo\labels
```

and then build the graph by running
```
python3 build_graph.py \path\to\feature\and\csv\files  \path\to\save\final\graph
```