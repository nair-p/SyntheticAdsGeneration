from urlextract import URLExtract
import validators
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import haversine as hs
from scipy.stats import entropy
from collections import Counter
import ast
from tqdm import tqdm

# FUNCTION THAT COUTNS THE NUMBER OF VALID, INVALID AND TOTAL URLS IN THE AD TEXT
def url_count(texts):
    num_invalid_urls = 0
    num_valid_urls = 0
    num_urls = 0
    extractor = URLExtract()
    for text in tqdm(texts):
        urls = extractor.find_urls(text)
        num_urls += len(urls)
        for url in urls:
            valid=validators.url(url)
            if valid:
                num_valid_urls += 1
            else:
                num_invalid_urls += 1
    return num_valid_urls, num_invalid_urls, num_urls


# FUNCTION THAT CALCULATES THE RADIUS OF LOCATIONS COVERED IN A CLUSTER
def find_loc_radii(list_locs):
    # this function finds the maximum radius covered by a given list of locations
    '''
    input: list_locs - list of locations as city names
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


# FUNCTION RETURNS THE ENTROPY OF PHONE NUMBERS
def find_entropy(numbers, base=None):
    number_freq = list(Counter(numbers).values())
    ent = entropy(number_freq, base=base)
    return ent


# FUNCTION RETURNS THE NUMBER OF ADS IN A WEEK
def get_num_ads_per_week(cluster, col_name='day_posted'):
    num_ads = 0
    ads_per_date = {}
    ads_per_week = []

    cluster[col_name] = pd.to_datetime(cluster[col_name], infer_datetime_format=True)
    dates = list(cluster[col_name].unique())
    dates.sort()
    for date, grp in cluster.groupby(col_name):
        ads_per_date[date] = grp.id.count()
    
    for i in tqdm(range(len(dates)-1)):
        d1 = pd.to_datetime(dates[i], infer_datetime_format=True)
        d2 = pd.to_datetime(dates[i+1], infer_datetime_format=True)
        if pd.isnull(d1) or pd.isnull(d2):
            continue

        day1 = (d1 - timedelta(days=d1.weekday()))
        day2 = (d2 - timedelta(days=d2.weekday()))
        num_weeks = (day2 - day1).days / 7
        if num_weeks == 0:
            ads_per_week.append(ads_per_date[d1] + ads_per_date[d2])
        else:
            ads_per_week.append(1)

    if len(ads_per_week) == 0:
        return 0.
    else:
        return max(ads_per_week)



