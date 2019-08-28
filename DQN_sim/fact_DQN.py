import fact_sim_n as fact_sim
import numpy as np
import matplotlib.pyplot as plt
import DeepQNet
from itertools import chain

sim_time = 5e5

# Dictionary where the key is the name of the machine and the value is [station, proc_t]
machine_dict = {'m0': 's1', 'm2': 's2', 'm1': 's1', 'm3': 's2'}

# recipes give the sequence of stations that must be processed at for the wafer of that head type to be completed
recipes = {"ht1": [["s1", 5, 0]], "ht2": [["s1", 5, 0], ["s2", 5, 0]]}

part_mix = [3, 3]

break_mean = 1e5

repair_mean = 20


def get_state(my_sim):
    # Calculate the state space representation.
    # This returns a list containing the number of parts in the factory for each combination of head type and sequence
    # step
    state_rep = [len([wafer for queue in my_sim.queue_lists.values() for wafer in queue if wafer.HT
                 == ht and wafer.seq == s]) for ht in list(my_sim.recipes.keys()) for s in
                 list(range(len(my_sim.recipes[ht]) + 1))]
    # b is a one-hot encoded list with length indicating which machine the next action will correspond to
    b = np.zeros(len(my_sim.machines_list))
    b[my_sim.machines_list.index(my_sim.next_machine)] = 1
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

# create the dqn_agent object with the appropriate lenght of state and action space
dqn_agent = DeepQNet.DQN(state_space_dim=len(state), action_space=action_space)
a = list(my_sim.station_HT_seq.values())
print(mach.station)
print(allowed_actions)
print(state)

order_count = 0

while my_sim.env.now < sim_time:
    print('action taken at: ')
    print(my_sim.env.now)
    action_time = my_sim.env.now

    #Choose the action using the output of the dqn_agent
    action = dqn_agent.choose_action(state, allowed_actions)
    # Run the simulation for one timestep based on the chosen action.
    my_sim.run_action(mach, action[0], action[1])

    # Record the machine, state, allowed actions and reward at the new time step
    next_mach = my_sim.next_machine
    next_state = get_state(my_sim)
    next_allowed_actions = my_sim.allowed_actions
    reward = my_sim.step_reward

    # rewards.append(reward)
    print(f"state: {state}")
    print(f"next state: {next_state}")

    # Save the example for later training
    dqn_agent.remember(state, action, reward, next_state, next_allowed_actions)

    if my_sim.order_completed:
        # After each order train the dqn_agent
        dqn_agent.replay()
        order_count += 1
        if order_count >= 10:
            # After every 10 orders update the target network and reset the order count
            dqn_agent.train_target()
            order_count = 0

    # record the information for use again in the next training example
    mach, allowed_actions, state = next_mach, next_allowed_actions, next_state
    print(state)
    print(reward)

# print(my_sim.t_between_orders)

# Plot the time taken to complete each order
plt.plot(my_sim.t_between_orders)
plt.show()
