import simpy
import numpy as np
import random

from .helper_functions import compute_travel_time, solve_bp, generate_random_location


# define the rider class
class Rider:
    def __init__(self, env, rider_id, location, active=1, speed=30):
        self.env = env
        self.id = rider_id
        self.initial_loc = location
        self.location = location
        self.active = active
        self.assignment_pipe = simpy.Store(env, capacity=1)

        # setting up metrics
        self.n_orders, self.travel_dist, self.travel_dist_nobundle, self.delivery_time = 0, 0, 0, []
        env.process(self.run())

    def run(self):
        while True:
            [o1, o2] = yield self.assignment_pipe.get()  # order objects
            n_orders = 1 if o1.id == o2.id else 2

            # current loc -> restaurant
            travel_d1, travel_t1 = compute_travel_time(self.location, o1.restaurant_loc)
            print(f"{self.env.now:2.2f}: Rider {self.id} --> restaurant (orders: {o1.id},{o2.id}; travel time: {travel_t1:.2f})")
            yield self.env.timeout(travel_t1)

            # update metrics
            self.travel_dist += travel_d1
            self.travel_dist_nobundle += travel_d1

            # restaurant -> 1st dropoff
            travel_d2, travel_t2 = compute_travel_time(o1.restaurant_loc, o1.dropoff)
            print(f"{self.env.now:2.2f}: Rider {self.id} --> 1st dropoff (orders: {o1.id},{o2.id}; travel time: {travel_t2:.2f})")
            yield self.env.timeout(travel_t2)

            # update metrics
            self.delivery_time.append(self.env.now - o1.placement_time)  # order 1 delivery time
            self.travel_dist += travel_d2
            self.travel_dist_nobundle += travel_d2
            self.n_orders += 1

            if n_orders > 1:
                # 1st dropoff -> 2nd dropoff
                travel_d3, travel_t3 = compute_travel_time(o1.dropoff, o2.dropoff)
                print(f"{self.env.now:2.2f}: Rider {self.id} --> 2nd dropoff (orders: {o1.id},{o2.id}; travel time: {travel_t3:.2f})")
                yield self.env.timeout(travel_t3)

                # update metrics
                self.delivery_time.append(self.env.now - o2.placement_time)  # order 2 delivery time
                self.travel_dist += travel_d3
                self.n_orders += 1

                travel_d4, _ = compute_travel_time(o1.dropoff, o2.restaurant_loc)
                travel_d5, _ = compute_travel_time(o2.restaurant_loc, o2.dropoff)
                self.travel_dist_nobundle += travel_d4 + travel_d5

            # wait 1 minute before being ready
            # self.wait(t=1)

            print(f"{self.env.now:2.2f}: Rider {self.id} delivery complete. Now active.")

            # update state
            self.location = o2.dropoff
            self.active = True

    def wait(self, t=1):
        yield self.env.timeout(t)

class Order:
    def __init__(self, env, order_id:int, placement_time, restaurant:list, dropoff, food_places):
        self.env = env
        self.id = order_id
        self.placement_time = placement_time
        self.restaurant = restaurant # [lat, long, prep_time]
        self.restaurant_loc = food_places[self.restaurant][:2]  # [lat, long]

        self.ready_time = food_places[self.restaurant][-1] + self.placement_time
        self.dropoff = dropoff  # [lat, long]


# define processes

def order_generator(env, pending_orders, order_list, ORDER_INTERARRIVAL, food_places):
    """ Generates orders based on defined inter-arrival time. """

    order_id = 0
    while True:
        interarrival_time = random.expovariate(1.0 / ORDER_INTERARRIVAL)  # average interarrival time (Poisson Distribution)
        yield env.timeout(interarrival_time)

        # randomize properties
        restaurant = random.choice(list(food_places.keys()))  # uniform distribution
        dropoff = generate_random_location()

        o = Order(env, order_id, placement_time=env.now, restaurant=restaurant, dropoff=dropoff)
        pending_orders.append(o)
        print(f"{env.now:2.2f}: Order {o.id} --- restaurant: {o.restaurant}, dropoff: ({o.dropoff[0]:.2f}, {o.dropoff[1]:.2f})")
        order_id += 1

        order_list.append(o)

def assignment_manager(env, riders, pending_orders, T=1):
    global ready_orders, assignment

    while True:
        yield env.timeout(T)

        # choose active riders and ready orders
        pending_orders_id = [o.id for o in pending_orders]
        ready_orders = [o for o in pending_orders]
        active_riders = [r.id for r in riders if r.active]


        if active_riders and ready_orders:

            print()
            print(f"{env.now:2.2f}: ========== Active riders: {active_riders} | Pending orders: {pending_orders_id} ================")
            # get rider-order assignment
            assignment = solve_bp(riders, ready_orders)  # output is an Nx2 matrix  (N riders, 2 orders)

            # assign orders
            remove_ords = []  # orders to remove from pending list
            for ridx, orders in enumerate(assignment):
                rider = riders[ridx]  # rider obj

                # get only first assigned order pair | set no assignent as None
                orders_nuniq = orders[0] if len(orders) > 0 else None  # [o1, o2]
                orders_uniq = np.unique(orders[0]).tolist() if len(orders) > 0 else None  #

                if orders_uniq:
                      ord_nuniq_id = [pending_orders_id.index(orders_nuniq[0]), pending_orders_id.index(orders_nuniq[1])]
                      rider.assignment_pipe.put(np.array(pending_orders)[ord_nuniq_id])  # assign order to rider object
                      rider.active = 0  # set rider activity status to inactive
                      remove_ords.extend(orders_uniq)  # add the index of orders to remove from pending orders

                      print(f"       Rider {ridx} assigned to orders {orders_uniq}")

            # remove assigned orders
            remove_ords.sort(reverse=True)
            for x in [o for o in pending_orders if o.id in remove_ords]:
                pending_orders.remove(x)

            # print("-"*90)
            print()
