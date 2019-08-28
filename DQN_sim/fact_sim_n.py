
# coding: utf-8

# In[5]:

import simpy
from collections import namedtuple, Counter
from itertools import count, filterfalse
import random

class wafer_box(object):
    def __init__(self, sim_inst, number_wafers, HT, wafer_index):
        self.env = sim_inst.env
        self.name = f"w{wafer_index}"
        self.start_time = sim_inst.env.now
        self.number_wafers = number_wafers
        self.HT = HT
        self.seq = 0

class Machine(object):
    def __init__(self, sim_inst, name, station, break_mean=None, repair_mean=None):
        self.env = sim_inst.env
        self.name = name
        self.station = station
        self.available = True
        self.broken = False
        self.wafer_being_proc = None
        self.parts_made = 0
        self.break_mean = break_mean
        if break_mean is not None:
            self.time_to_fail = self.time_to_failure()

        self.process = None
        self.repair_mean = repair_mean

    def time_to_failure(self):
        """Return time until next failure for a machine."""
        return random.expovariate(1/self.break_mean)

    def time_to_repair(self):
        """Return time until next failure for a machine."""
        return random.expovariate(1/self.repair_mean)

    def break_machine(self):
        """Break the machine after break_time"""
        assert not self.broken
        start = self.env.now
        try:
            yield self.env.timeout(self.time_to_fail)
            self.process.interrupt()
            self.time_to_fail = self.time_to_failure()
        except:
            self.time_to_fail -= self.env.now-start

    def part_process(self, wafer, sim_inst):
        # This function defines a process where a part of head type HT and sequence step seq is processed on the machine

        # get the amount of time it takes for the operation to run
        proc_t = sim_inst.recipes[wafer.HT][wafer.seq][1]*wafer.number_wafers + sim_inst.recipes[wafer.HT][wafer.seq][2]

        done_in = proc_t
        while done_in:
            try:
                if self.break_mean is not None:
                    break_process = self.env.process(self.break_machine())
                start = self.env.now
                print("started processing wafer %s on machine %s at %s"%(wafer.name, self.name, start))
                # wait until the process is done
                yield sim_inst.env.timeout(done_in)
                # set the wafer being processed to None
                self.wafer_being_proc = None
                # set machine to be available to process part
                self.available = True
                print("Completed the process step of wafer %s on machine %s at %s and sent to "
                      "next machine."%(wafer.name, self.name, self.env.now))
                # set the wafer to be at the next step in the sequence
                wafer.seq += 1
                # if seq is not the last sequence step then find the next station and choose actions for each of the
                # available machines in that station
                if wafer.seq < (len(sim_inst.recipes[wafer.HT])):
                    # add the part to the corresponding queue for the next operation in the sequence
                    sim_inst.queue_lists[sim_inst.recipes[wafer.HT][wafer.seq][0]].append(wafer)
                else:
                    # add the part to the list of completed parts
                    sim_inst.queue_lists['complete'].append(wafer)
                    print("Finished processing wafer %s at %s"%(wafer.name, self.env.now))
                    # get a list of the number of wafers for each head type that are completed
                    num_wafers_per_HT = [len([wafer for wafer in sim_inst.queue_lists['complete'] if wafer.HT == ht]) for ht in list(sim_inst.recipes.keys())]
                    # Get the next order in the order list
                    next_order = sim_inst.order_list[0]

                    # if there are enough wafers of each head type to fulfill the order then remove them from the
                    # completed list and remove the order from the order list
                    if all([ht_num >= next_order[ht_ind] for ht_ind, ht_num in enumerate(num_wafers_per_HT)]):
                        sim_inst.order_completed = True
                        sim_inst.t_between_orders.append(sim_inst.env.now-sim_inst.order_complete_time)
                        sim_inst.step_reward += 100000/(sim_inst.env.now-sim_inst.order_complete_time)
                        sim_inst.order_complete_time = sim_inst.env.now

                        print("FINISHED A COMPLETE ORDER AT %s"%(sim_inst.env.now))
                        print("************************************************************************************"
                              "***********")
                        for ht, num in next_order._asdict().items():
                            sim_inst.queue_lists['complete'] = list(filterfalse(lambda wafer, c=count(): wafer.HT == ht and next(c) < num, sim_inst.queue_lists['complete']))
                        del sim_inst.order_list[0]

                        if len(sim_inst.order_list) <= 0:
                            sim_inst.order_list.extend([sim_inst.weekly_order._make(sim_inst.part_mix) for i in range(1)])
                            #Add all the parts contained in the series of orders to the queues of the stations for
                            # their first processing steps
                            print("################## Add a new order list to the queue ##################")
                            for week_order in [sim_inst.weekly_order._make(sim_inst.part_mix) for i in range(1)]:
                                for HT, num in week_order._asdict().items():
                                    for i in range(num):
                                        sim_inst.queue_lists[sim_inst.recipes[HT][0][0]].append(
                                            wafer_box(self, 4, HT, sim_inst.wafer_index))
                                        sim_inst.wafer_index += 1
                if self.break_mean is not None:
                    break_process.interrupt()
                done_in = 0

            except simpy.Interrupt:
                self.broken = True
                done_in -= self.env.now - start
                yield self.env.timeout(self.time_to_repair())
                self.broken = False

        # Parts completed by this machine
        self.parts_made += 1


    def get_allowed_actions(self, sim_inst):
        #find all (HT, seq) tuples with non zero queues at the station of this machine
        return sorted(list(set((wafer.HT, wafer.seq) for wafer in sim_inst.queue_lists[self.station])))

class FactorySim(object):
    #Initialize simpy environment and set the amount of time the simulation will run for
    def __init__(self, sim_time, m_dict, recipes, part_mix, break_mean=None, repair_mean=None):
        self.break_mean = break_mean
        self.repair_mean = repair_mean
        self.order_completed = False
        self.allowed_actions = None
        self.env = simpy.Environment()
        self.Sim_time = sim_time
        self.next_machine = None
        # self.machine_failure = False
        # Initialize an index that will be used to name each wafer box
        self.wafer_index = 0

        # Dictionary where the key is the name of the machine and the value is [station, proc_t]
        self.machine_dict = m_dict

        self.machines_list = [Machine(self, mach[0], mach[1], self.break_mean, self.repair_mean) for mach in self.machine_dict.items()]

        # create a list of all the station names
        self.stations = list(set(list(self.machine_dict.values())))

        # sim_inst.recipes give the sequence of stations that must be processed at for the wafer of that head type to be completed
        self.recipes = recipes

        # create a named tuple to represent the number of parts of each head type that will be produced each week
        self.weekly_order = namedtuple('weekly_order', list(self.recipes.keys()))

        self.part_mix = part_mix

        # create a list of orders to represent a series of orders that will be due each week
        self.order_list = [self.weekly_order._make(self.part_mix) for i in range(1)]

        # Create a dictionary which holds lists that will contain the queues of wafer_box objects at each station and that have
        # been completed
        self.queue_lists = {station: [] for station in self.stations}
        self.queue_lists['complete'] = []

        self.order_complete_time = 0
        self.t_between_orders = []
        self.step_reward = 0


        # Creates a dictionary where the key is the toolset name and the value is a list of tuples of all head type and
        # sequence step combinations which may be processed at that station
        self.station_HT_seq = {station: [] for station in self.stations}

        for HT in self.recipes.keys():
            for seq, step in enumerate(self.recipes[HT]):
                self.station_HT_seq[step[0]].append((HT, seq))

        # Add all the parts contained in the series of orders to the queues of the stations for their first processing steps
        for week_order in self.order_list:
            for HT, num in week_order._asdict().items():
                for i in range(num):
                    self.queue_lists[self.recipes[HT][0][0]].append(wafer_box(self, 4, HT, self.wafer_index))
                    self.wafer_index += 1
        self.number_of_machines = len(self.machine_dict)

    def start(self):
        for machine in self.machines_list:
            if machine.available:
                allowed_actions = machine.get_allowed_actions(self)
                if len(allowed_actions) > 0:
                    self.next_machine = machine
                    self.allowed_actions = allowed_actions
                    return

    def run_action(self, machine, ht, seq):
        self.order_completed = False
        self.step_reward = 0
        # Set the machine to be unavailable to process parts because it is now busy
        assert machine.available
        machine.available = False
        # Find the wafer that has that HT and seq
        wafer_choice = next(
            wafer for wafer in self.queue_lists[machine.station] if wafer.HT == ht and wafer.seq == seq)
        #             print("Wafer chosen to process on Machine %s is %s"%(machine.name, wafer_choice.name))
        # print("Wafer %s chosen for machine %s for action %s from list %s at time %s" % (
        # wafer_choice.name, machine.name, (ht, seq), allowed_actions, env.now))
        # set the wafer being processed on this machine to wafer_choice
        machine.wafer_being_proc = wafer_choice
        # Remove the part from it's queue
        self.queue_lists[machine.station].remove(wafer_choice)
        # Begin processing the part on the machine
        machine.process = self.env.process(machine.part_process(wafer_choice, self))

        for machine in self.machines_list:
            if machine.available:
                allowed_actions = machine.get_allowed_actions(self)
                if len(allowed_actions) > 0:
                    self.next_machine = machine
                    self.allowed_actions = allowed_actions
                    return

        while True:
            self.env.step()
            for machine in self.machines_list:
                if machine.available:
                    allowed_actions = machine.get_allowed_actions(self)
                    if len(allowed_actions) > 0:
                        self.next_machine = machine
                        self.allowed_actions = allowed_actions
                        return


