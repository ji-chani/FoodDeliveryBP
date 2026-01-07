import geopandas as gpd
import simpy
import random
import json
import networkx as nx
import numpy as np


from modules import Rider, order_generator, assignment_manager, generate_random_location

#Load Road Network and Food Place Data

node_network = gpd.read_file(f"resources/NamedNodesLBOnly.shp")
nodes_wgs84 = node_network.to_crs(epsg=4326)
road_network = gpd.read_file(f"resources/NamedRoadsLBOnly.shp")
food_network = gpd.read_file(f"resources/Food Places.json")
food_wgs84 = food_network.to_crs(epsg=4326)

# Create graph
G = nx.Graph()

# Add nodes to graph
for idx, row in node_network.iterrows():
    G.add_node(row['node_id'], pos=(row.geometry.x, row.geometry.y))

# Add edges with weights = road length
for idx, row in road_network.iterrows():
    length = row.geometry.length
    G.add_edge(row['start_id'], row['end_id'], weight=length, road_id=row['road_id'])


# Preparing restaurants and their properties

#Name Cleaning
food_network["location"] = food_network["location"].astype(str)
food_network["location"] = food_network["location"].str.strip("'")

#Extract Name Only
food_network["location"] = food_network["location"].apply(json.loads)
food_network["place_name"] = food_network["location"].apply(lambda x: x["name"])

#Create dictionary
randomize_prep = False
food_places = {
    row["place_name"]: [row["geometry"].y, row["geometry"].x, round(random.uniform(10,30),2) if randomize_prep else 15]  # [lat, long, prep_time]
    for _, row in food_network.iterrows()
}

# ---- Main Implementation

RANDOM_SEED = 21  # for reproduction purposes
N = 5  # no. of riders
OPERATION_TIME = 24 * 60   # total no. of minutes the food delivery system operates
ORDER_INTERARRIVAL = 3.0  # average interarrival time of orders (Poisson Distribution) | interpretation: an order arrives every ORDER_INTERARRIVAL minutes on average
ASSIGNMENT_INTERVAL = 1  # no. of mins at which rider-order assignment is computed

random.seed(RANDOM_SEED)
env = simpy.Environment()
pending_orders, order_list = [], []
riders = [Rider(env, i, location=generate_random_location(near="restaurant"), active=1) for i in range(N)]

env.process(order_generator(env, pending_orders, order_list, ORDER_INTERARRIVAL))
env.process(assignment_manager(env, riders, pending_orders))

env.run(until=OPERATION_TIME)  # in minutes


print(f"Simulation Complete! \n Duration: {OPERATION_TIME/60} hrs \n Total riders: {N} \n Total orders: {len(order_list)} \n Pending orders: {len(pending_orders)}")
print()

print(f"Total orders delivered: {[r.n_orders for r in riders]}")
print(f"Total travel distance (m): {[round(r.travel_dist,2) for r in riders]}")
print(f"bundle-unbundled distance (m) variance: {[round(r.travel_dist_nobundle - r.travel_dist,5) for r in riders]}")
print(f"Average delivery time (min) per order: {[round(float(np.mean(r.delivery_time)),2) for r in riders]}")
