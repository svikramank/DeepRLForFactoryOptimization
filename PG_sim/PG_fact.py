import fact_simno1act as fact_sim
import numpy as np
import matplotlib.pyplot as plt
import PG_Class
from itertools import chain

sim_time = 1e6

# Dictionary where the key is the name of the machine and the value is [station, proc_t]
machine_dict = {'m0': 's1', 'm2': 's2', 'm1': 's1', 'm3': 's2'}

# recipes give the sequence of stations that must be processed at for the wafer of that head type to be completed
recipes = {"ht1": [["s1", 5, 0]], "ht2": [["s1", 5, 0], ["s2", 5, 0]]}

part_mix = [3, 3]

break_mean = 1e4

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
# create the pol_grad object with the appropriate lenght of state and action space
pol_grad = PG_Class.PolGrad(action_space, len(state))

episode_states, episode_actions, allRewards, episode_allowed_a = [],[],[],[]

print(mach.station)
print(allowed_actions)
print(state)


while my_sim.env.now < sim_time:
    episode_states.append(state)
    episode_allowed_a.append(allowed_actions)
    
    action = pol_grad.choose_action(state, allowed_actions)
    action_ = np.zeros(action_size)
    action_[action] = 1
    episode_actions.append(action_)
    
    action = action_space[action]
    
    if my_sim.order_completed:
        # Calculate discounted reward
        episode_rewards_ = np.ones(np.asarray(episode_states).shape[0])
        episode_rewards_ *= my_sim.step_reward
        pol_grad.train_target(np.asarray(episode_states), np.asarray(episode_actions), episode_rewards_,
                              episode_allowed_a)
        
        # Reset the transition stores
        episode_states, episode_actions, episode_allowed_a = [],[],[]

    my_sim.run_action(mach, action[0], action[1])
    state = get_state(my_sim)
    allowed_actions = my_sim.allowed_actions
    mach = my_sim.next_machine
    print(my_sim.order_completed)
    print(state)
    print(my_sim.step_reward)

# print(my_sim.t_between_orders)

plt.plot(my_sim.t_between_orders)
plt.show()
