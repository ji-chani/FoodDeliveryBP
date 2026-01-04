from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, lpSum
from shapely.geometry import Point
import networkx as nx
import numpy as np
import random



def closest_node(nodes_wgs84, coords) -> str:
  """ computes for the nearest network node to a location coordinates.

  :param: coords (tuple): location in (latitude, longitude) ex: (14.179278667538483, 121.2392308566381)
  """

  start_location = Point(coords[1], coords[0])

  sindex = nodes_wgs84.sindex

  # Get name of nearest node
  nearest_index = list(sindex.nearest(start_location, return_all=False))[1]

  nearest_node_id = nodes_wgs84.iloc[nearest_index]['node_id']
  return nearest_node_id.item()


#returns path (nodes) and path length
def calculate_path_length(G, locA, locB) -> tuple:
  """ calculates the path nodes and path length between two given locations: locA, locB.

  :param: locA / locB (str): start / end location index names. Ex: ABC, MNO
  :return: path_nodes (list), path_length (float)
  """

  path_nodes = nx.shortest_path(G, source=locA, target=locB, weight='weight', method='dijkstra')
  path_length = nx.shortest_path_length(G, source=locA, target=locB, weight='weight', method='dijkstra')
  return path_nodes, path_length


def compute_travel_time(G, origin:tuple, destination:tuple, speed=20):
    """ Compute travel time from origin to destination given a speed """
    orig, dest = closest_node(origin), closest_node(destination)  # get closest network nodes to origin and destination
    path_length = nx.shortest_path_length(G, orig, dest, weight='weight')  # compute for path length of orig -> dest
    return path_length, path_length / 1000 / speed * 60

def generate_random_location(nodes_wgs84, food_wgs84, near=None, mu=0, std=0.0001):
    """ Generate random location within the promiximity of existing road network.

    :param: near: "restaurant" or None
    """

    if near == "restaurants":
        wgs84 = food_wgs84
    else:
        wgs84 = nodes_wgs84

    random_idx = random.randint(0,len(wgs84)-1)  # random location chosen from current network nodes
    loc = (wgs84['geometry'][random_idx].y, wgs84['geometry'][random_idx].x)
    loc = np.array(loc)
    noise = np.random.normal(mu, std, size=loc.shape)  # gaussian noise
    new_loc = loc + noise
    return (float(new_loc[0]), float(new_loc[1]))


def cost_matrix(G, active_riders:list, ready_orders:list) -> np.ndarray:
    N, M = len(active_riders), len(ready_orders)

    # get prerequisites
    riders_loc = [closest_node(r.location) for r in active_riders]
    riders_act = np.diag([r.active for r in active_riders])
    rest_loc = [closest_node(o.restaurant_loc) for o in ready_orders]
    dropoff_loc = [closest_node(o.dropoff) for o in ready_orders]

    # rider to restaurant distances
    src_dest = [(s, d) for s in riders_loc for d in rest_loc]
    rider_rest_dist = np.array([nx.shortest_path_length(G, sd[0], sd[1], weight="weight") for sd in src_dest]).reshape(N, M)

    # restaurant to 1st dropoff distances
    src_dest = [(s, d) for s,d in zip(rest_loc, dropoff_loc)]
    rest_drop1_dist = np.array([nx.shortest_path_length(G, sd[0], sd[1], weight="weight") for sd in src_dest])
    rest_drop1_dist = np.resize(rest_drop1_dist, N*M).reshape(N, M)  # repeat and resize for later combination

    # 1st to 2nd dropoff distances
    src_dest = [(s, d) for s in dropoff_loc for d in dropoff_loc]
    drop1_drop2_dist = np.array([nx.shortest_path_length(G, sd[0], sd[1], weight="weight") for sd in src_dest]).reshape(M, M)
    np.fill_diagonal(drop1_drop2_dist, 0)  # making sure that distance to same loc is 0

    # combine distances

    # rider -> 1st dropoff
    rider_drop1_dist = rider_rest_dist + rest_drop1_dist

    # rider -> 1st dropoff -> 2nd dropoff
    indices = [(i,j,k) for i in range(N) for j in range(M) for k in range(M)]
    rider_drop2_dist = np.array([rider_drop1_dist[i][j]+drop1_drop2_dist[j][k] for (i,j,k) in indices]).reshape(N, M**2)

    return rider_drop2_dist

def solve_bp(riders:list, ready_orders:list):
    """ Compute for the optimal rider-order assignment using BP """

    # initial required parameters
    N, M = len(riders), len(ready_orders)
    pairs = [[j,k] for j in range(M) for k in range(M)]
    same_rest_ind = np.array([True if o1.restaurant == o2.restaurant else False for o1 in ready_orders for o2 in ready_orders])
    riders_act = np.array([r.active for r in riders])
    indices = {k: [index for index,p in enumerate(pairs) if k in p] for k in range(M)}  # get column indices where key is observed

    # define problem
    prob = LpProblem("Rider_Order_Assignment", LpMinimize)

    # binary variables
    vars = np.array([LpVariable(f"x{i}_{j}", 0, 1, cat="Binary") for i in range(N) for j in range(M**2)]).reshape(N,M**2)

    # Calculate the cost matrix
    costs = cost_matrix(riders, ready_orders)

    # objective function: sum product of cost matrix and variables
    prob += lpSum(costs * vars), "Objective Function"

    # constraint 1: each order assigned exactly once
    for k in range(M):
        prob += lpSum(vars[i,indices[k]] for i in range(N)) == 1, f"order{k}_assigned_once"

    # constraint 2: each rider can take at most one pair (capacity constraint)
    for i in range(N):
        prob += lpSum(vars[i,k] for k in range(M**2)) <= 1, f"rider{i}_capacity"

    # constraint 3 & 4: bundling feasibility and rider activity feasibility
    for i in range(N):
        for k in range(M**2):
            prob += vars[i,k] <= same_rest_ind[k], f"bundling_feasibility_{i}_{k}"

            prob += vars[i,k] <= riders_act[i], f"rider_activity_{i}_{k}"

    # solve the problem
    prob.solve()

    # status and optimal objective value
    # print("   Status:", LpStatus[prob.status])
    # print("   Optimal Objective Value:", value(prob.objective))

    # extract rider-order assignments
    optimal_var_matrix = np.array([[vars[i][j].varValue for j in range(M**2)] for i in range(N)])
    assigned_orders = []
    for i in range(N):
        assigned_idx = np.where(optimal_var_matrix[i,:] == 1)[0]
        orders_idx = [pairs[idx] for idx in assigned_idx]
        orders = [[ready_orders[o[0]].id, ready_orders[o[1]].id] for o in orders_idx]
        assigned_orders.append(orders)

    return assigned_orders

