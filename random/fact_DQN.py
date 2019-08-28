import fact_sim_n as fact_sim
import numpy as np
import pandas as pd
import math 
import matplotlib
import random
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from itertools import chain

sim_time = 5e5
WEEK = 24*7*60
NO_OF_WEEKS = math.ceil(sim_time/WEEK)

recipes = pd.read_csv("/Users/vikramanksingh/Downloads/recipes.csv")
machines = pd.read_csv("/Users/vikramanksingh/Downloads/machines.csv")

# Create the machine dictionary (machine:station)
machine_d = dict()
for index, row in machines.iterrows():
    d = {row[0]:row[1]}
    machine_d.update(d)

# Modifying the above list to match the stations from the two datasets 
a = machines.TOOLSET.unique()
b = recipes.TOOLSET.unique()
common_stations = (set(a) & set(b))
ls = list(common_stations)

# This dictionary has the correct set of stations
modified_machine_dict = {k:v for k,v in machine_d.items() if v in ls}

# Removing unncommon rows from recipes 
for index, row in recipes.iterrows():
    if row[2] not in ls:
        recipes.drop(index, inplace=True)

recipes = recipes.dropna()
recipe_dict = dict()
for ht in list(recipes.HT.unique()):
    temp = recipes.loc[recipes['HT'] == ht]
    if len(temp) > 1:
        ls = []
        for index, row in temp.iterrows():
            ls.append([row[2], row[3], row[4]])
        d  = {ht:ls}
        recipe_dict.update(d)
    else:
        ls = []
        ls.append([row[2], row[3], row[4]])
        d = {ht:ls}
        recipe_dict.update(d)


# weekly quota for head types
head_types = recipes.keys()
quota_dict = {}

for ht in head_types:
    d = {ht:random.randint(20,50)}
    quota_dict.update(d)

# Dictionary where the key is the name of the machine and the value is [station, proc_t]
# machine_dict = {'m0': 's1', 'm2': 's2', 'm1': 's1', 'm3': 's2'}
machine_dict = modified_machine_dict

# recipes give the sequence of stations that must be processed at for the wafer of that head type to be completed
# recipes = {"ht1": [["s1", 5, 0]], "ht2": [["s1", 5, 0], ["s2", 5, 0]]}
recipes = recipe_dict

# part_mix = [3, 3]
part_mix = [30 for _ in range(len(recipes.keys()))]


break_mean = 1e5

repair_mean = 20


def get_state(sim):
    # Calculate the state space representation.
    # This returns a list containing the number of parts in the factory for each combination of head type and sequence
    # step
    state_rep = [len([wafer for queue in sim.queue_lists.values() for wafer in queue if wafer.HT
                 == ht and wafer.seq == s]) for ht in list(sim.recipes.keys()) for s in
                 list(range(len(sim.recipes[ht]) + 1))]
    # b is a one-hot encoded list with length indicating which machine the next action will correspond to
    b = np.zeros(len(sim.machines_list))
    b[sim.machines_list.index(sim.next_machine)] = 1
    state_rep.extend(b)
    return state_rep

def choose_action(state, allowed_actions):
    return random.choice(allowed_actions)


# Create the factory simulation object
my_sim = fact_sim.FactorySim(sim_time, machine_dict, recipes, part_mix)
# start the simulation
my_sim.start()
# Retrieve machine object for first action choice
mach = my_sim.next_machine
# Save the state and allowed actions at the start for later use in training examples
state = get_state(my_sim)
allowed_actions = my_sim.allowed_actions
# The action space is a list of tuples of the form [('ht1',0), ('ht1',1), ..., ('ht2', 0), ...] indicating the head
# types and sequence steps for all allowed actions.
action_space = list(chain.from_iterable(my_sim.station_HT_seq.values()))
action_size = len(action_space)

# print("state:", state)
# print("allowed_actions:", allowed_actions)
# print("action_space:", action_space)
# print("action size:", action_size)


while my_sim.env.now < sim_time:
    action = choose_action(state, allowed_actions)

    my_sim.run_action(mach, action[0], action[1])

    # Record the machine, state, allowed actions and reward at the new time step
    next_mach = my_sim.next_machine
    next_state = get_state(my_sim)
    next_allowed_actions = my_sim.allowed_actions
    reward = my_sim.step_reward

    print(f"state: {state}")
    print(f"next state: {next_state}")

    # record the information for use again in the next training example
    mach, allowed_actions, state = next_mach, next_allowed_actions, next_state
    print(state)
    print(reward)

# Plot the time taken to complete each order
plt.plot(my_sim.t_between_orders)
plt.show()










































