# SyntheticAdsGeneration
Code for adding M.Os to synthetically generated ads using GPT3.5

This code-base assumes that the starter files (in csv format) contains a corpus of synthetically generated ads and follows the below-mentioned pipeline for adding specific kinds of activities. 

First, we want to add posting date, location, social media tags and phone numbers.
For extracting names, we use [HTNER](www.github.com/HTNER). 