import json
import os

from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from shapely import Polygon, Point

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

from scipy.stats.contingency import association


source2filename = {"crewed":"./data/crewed_train.json",
				   "suas":"./data/suas_train.json",
				   "satellite":"./data/satellite_train.json"}

source2title = {"crewed":"Crewed Aircraft",
				"suas":"Drone",
				"satellite":"Satellite",
				"max_all":"Max All Sources",
				"agree_all":"All Sources That Agree", 
				"closest_all": "All Sources That Are Closet to Event Date",
				"farthest_all": "All Sources That Are Farthest to Event Date"}

def pick_agree_label(labels):
	l = list(set(labels))
	if len(l) == 1:
		return l[0]
	return None

def pick_agree_jds_label(labels):
	return pick_agree_label(labels)

def pick_agree_ground_label(labels):
	return pick_agree_label(labels)

def pick_max_ground_label(labels):
	if 'Destroyed' in labels:
		return 'Destroyed' 
	if 'Major' in labels:
		return 'Major' 
	if 'Minor' in labels:
		return 'Minor' 
	if 'Affected' in labels:
		return 'Affected' 
	if 'Unaffected' in labels:
		return 'Unaffected' 
	if 'Inaccessible' in labels:
		return 'Inaccessible'
	print("Returning none", labels)
	return None

def pick_max_jds_label(labels):
	if "destroyed" in labels:
		return "destroyed"
	if "major damage" in labels:
		return "major damage"
	if "minor damage" in labels:
		return "minor damage"
	if "no damage" in labels:
		return "no damage"
	if "un-classified" in labels:
		return "un-classified"
	if "obscured" in labels:
		return "obscured"
	print("Returning None", labels)
	return None

def pick_closest_jds_label(dates):
	return min(datetime.strptime(date, "%m/%d/%Y") for date in dates)

def pick_farthest_jds_label(dates):
	return max(datetime.strptime(date, "%m/%d/%Y") for date in dates)

def pick_temporal_jds_label(mapped_labels, strategy):
	date = None
	if strategy == "closest":
		date_index = [index for index, date in enumerate(mapped_labels["jds_creation_date"]) if datetime.strptime(date, "%m/%d/%Y") == pick_closest_jds_label(mapped_labels["jds_creation_date"])]
	if strategy == "farthest":
		date_index = [index for index, date in enumerate(mapped_labels["jds_creation_date"])  if datetime.strptime(date, "%m/%d/%Y") == pick_farthest_jds_label(mapped_labels["jds_creation_date"])]

	return [mapped_labels["crasar_u_droids"][i] for i in date_index] 

def parse_ground_level_labels(path_to_ground_level_labels):
	with open(path_to_ground_level_labels, "r") as f:
		result = json.loads(f.read())
		return result

def parse_crasar_u_droids_data(path_to_crasar_u_droids_labels):
	with open(path_to_crasar_u_droids_labels, "r") as f:
		result = json.loads(f.read())
		return result

def parse_crasar_u_droids_statistics(path_to_crasar_u_droids_statistics):
	return pd.read_csv(path_to_crasar_u_droids_statistics)

def get_label_mappings(residential_ground_level_data, commerical_ground_level_data, crasar_u_droids_data, aerial_label_statistics, neighbors_count):

	ground_level_features = residential_ground_level_data["features"] + commerical_ground_level_data["features"]
	ground_level_coords = []
	for entry in ground_level_features:
		ground_level_coords.append([entry["geometry"]["coordinates"][1], entry["geometry"]["coordinates"][0]])

	neigh = NearestNeighbors(n_neighbors=neighbors_count, radius=0.001)
	neigh.fit(ground_level_coords)

	mapped_labels = defaultdict(lambda:{"ground":[], "crasar_u_droids":[], "jds_creation_date":[]})

	for boundary_id, payload in crasar_u_droids_data.items():
		for building in payload:
			ortho_title = building["filename"].replace(".json", "")
			if aerial_label_statistics[aerial_label_statistics['Orthomosaic'] == ortho_title]["Pre/Post Event"].iloc[0] == "POST":
				coords = [(p["lat"], p["lon"]) for p in building["EPSG:4326"]]
				building_polygon = Polygon(coords)
				building_polygon_centroid = building_polygon.centroid
				
				neighbors = neigh.kneighbors([[building_polygon_centroid.x, building_polygon_centroid.y]], neighbors_count, return_distance=False)
				
				for neighbor in neighbors[0]:
					ground_coord = ground_level_features[neighbor]["geometry"]["coordinates"]
					candidate_point = Point(ground_coord[1], ground_coord[0])

					if building_polygon.contains(candidate_point):
						mapped_labels[building["id"]]["ground"].append(ground_level_features[neighbor]["properties"]["DamageLevel"])
						mapped_labels[building["id"]]["crasar_u_droids"].append(building["label"])
						mapped_labels[building["id"]]["jds_creation_date"].append(aerial_label_statistics[aerial_label_statistics['Orthomosaic'] == ortho_title]["Date (mm/dd/yyy)"].iloc[0])

	return mapped_labels

def combine_sources(mapped_labels_dict):
	mapped_labels = defaultdict(lambda:{"sources":[], "ground":[], "crasar_u_droids":[], "jds_creation_date":[]})
	for source, payload in mapped_labels_dict.items():
		for key, labels in payload.items():
			mapped_labels[key]["ground"].extend(labels["ground"])
			mapped_labels[key]["crasar_u_droids"].extend(labels["crasar_u_droids"])
			mapped_labels[key]["sources"].append(source)
			mapped_labels[key]["jds_creation_date"].extend(labels["jds_creation_date"])
	return mapped_labels

def generate_table_data(mapped_labels, jds_strategy="max", ground_strategy="max"):
	all_ground_labels = []
	all_jds_labels = []
	for entry in mapped_labels.values():
		all_ground_labels.extend(entry["ground"])
		all_jds_labels.extend(entry["crasar_u_droids"])


	ground_labels = list(set(all_ground_labels))
	jds_labels = list(set(all_jds_labels))

	result = {}
	for jds_label in jds_labels:
		result[jds_label] = {}
		for ground_label in ground_labels:
			result[jds_label][ground_label] = 0

	for entry in mapped_labels.values():
		jds_key = None
		ground_key = None
		if jds_strategy == "max":
			jds_key = pick_max_jds_label(entry["crasar_u_droids"])
		elif jds_strategy == "agree":
			jds_key = pick_agree_jds_label(entry["crasar_u_droids"])
		elif jds_strategy == "closest":
			jds_key = pick_temporal_jds_label(entry, jds_strategy)
		elif jds_strategy == "farthest":
			jds_key = pick_temporal_jds_label(entry, jds_strategy)
		if ground_strategy == "max":
			ground_key = pick_max_ground_label(entry["ground"])
		if ground_strategy == "agree":
			ground_key = pick_agree_ground_label(entry["ground"])

		if jds_strategy in ["closest", "farthest"]:
			for k in jds_key:
				if ground_key and k:
					result[k][ground_key] += 1
		else:
			if ground_key and jds_key:
				result[jds_key][ground_key] += 1

	return result

def plot_confusion_matrix(table_data, source):
	df_cm = pd.DataFrame(table_data)
	df_cm = df_cm.loc[['Unaffected', 'Affected', 'Minor', 'Major', 'Destroyed'], ["no damage", "minor damage", "major damage", "destroyed"]]
	plt.figure(figsize = (10,7))
	sn.heatmap(df_cm, annot=True, cmap='RdBu_r', fmt=",d", vmin=0, vmax=1900)
	plt.title("Ground Level vs " + source2title[source] + " Damage Labels\nHurricane Ian | N="+str(df_cm.sum().sum()))
	plt.xlabel(source2title[source] + " Imagery Derived Label (Joint Damage Scale)")
	plt.ylabel("Ground Level Label (HAZUS-MH)")
	plt.savefig("ground_vs_" + source + ".png")

def compute_association(table_data, method):
	listed_table_data = []
	for key1 in table_data.keys():
		listed_table_data.append([])
		for key2 in table_data[key1].keys():
			listed_table_data[-1].append(table_data[key1][key2])

	data_for_association = np.array(listed_table_data)
	return association(data_for_association, method=method)

source = "crewed"

print("Parsing ground level labels...")
residential_ground_level_labels = parse_ground_level_labels("./data/lee_county_residential_damage_assessments.geojson")
commercial_ground_level_labels = parse_ground_level_labels("./data/lee_county_commercial_damage_assessments.geojson")
print("Parsing CRASAR-U-DROIDs statistics file...")
crasar_u_droids_stats = parse_crasar_u_droids_statistics("./data/statistics.csv")

crasar_u_droids_labels = {}
combined_mapped_labels = {}
for source in ["suas", "crewed", "satellite"]:
	print("Inspecting", source)
	crasar_u_droids_labels[source] = parse_crasar_u_droids_data("./" + source2filename[source])
	combined_mapped_labels[source] = get_label_mappings(residential_ground_level_labels, commercial_ground_level_labels, crasar_u_droids_labels[source], crasar_u_droids_stats, 10)
	table_data = generate_table_data(combined_mapped_labels[source])

	print("\tCramer's V                       ", compute_association(table_data, "cramer"))
	print("\tPearson’s Contingency Coefficient", compute_association(table_data, "pearson"))
	print("\tTschuprow's T                    ", compute_association(table_data, "tschuprow"))
	print("\tPlotting confusion matrix...")
	plot_confusion_matrix(table_data, source)

print("Combining sources to one map")
mapped_combined = combine_sources(combined_mapped_labels)

print("Generating table data...")
table_data = generate_table_data(mapped_combined)

print("Plotting confusion matrix...")
plot_confusion_matrix(table_data, "max_all")

print("Generating table data...")
table_data = generate_table_data(mapped_combined, jds_strategy="agree")

print("Plotting confusion matrix...")
plot_confusion_matrix(table_data, "agree_all")


print("Generating table data...")
table_data = generate_table_data(mapped_combined, jds_strategy="closest")

print("Plotting confusion matrix...")
plot_confusion_matrix(table_data, "closest_all")

print("Generating table data...")
table_data = generate_table_data(mapped_combined, jds_strategy="farthest")

print("Plotting confusion matrix...")
plot_confusion_matrix(table_data, "farthest_all")

print("Done!")
