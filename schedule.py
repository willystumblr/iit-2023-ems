import pandas as pd
import pulp as pl
import argparse
import numpy as np
from typing import Any 
import math

ELEC_PRICE = [38.8]*9+[53.0]+[72.8]*2+[53.0]+[72.8]*4+[53.0]*6+[38.8]
LOAD_PATH = "load/predict_for_0901.xlsx"
PV_PATH="pv_pred/summed.csv"


parser = argparse.ArgumentParser(description="Scheduler Mode")
parser.add_argument('--mode', type=str, 
                    default='test',
                    help="'test' for test with arbitary input, 'eval' for actual evaluation.")

args = parser.parse_args()


# Define the problem
prob = pl.LpProblem("ESS_Scheduling", pl.LpMinimize)

def define_problem(prob: Any, 
                   electricity_price: list,  
                   load: np.ndarray, 
                   pv_gen: np.ndarray, 
                   max_capacity: float=750.0, 
                   efficiency: float=0.93,
                   ) : 
    """Define Linear programming example

    Args:
        `max_capacity` (int, optional): Maximum battery capacity (kWh). Defaults to 750kWh.
        `efficiency` (float, optional): Charging/discharging efficiency. Defaults to 0.93
        `electricity_price` (list): Electricity price for each hour
        `load` (np.ndarray): Predicted load each hour
        `pv_gen` (np.ndarray): Predicted PV generation for each hour
    """
    
    
    # Parameters
    num_hours = 24
    
    # Variables
    energy_bought = pl.LpVariable.dicts("energy_bought", range(num_hours), 0)
    energy_sold = pl.LpVariable.dicts("energy_sold", range(num_hours), 0)
    energy_charged = pl.LpVariable.dicts("energy_charged", range(num_hours), 0)
    energy_discharged = pl.LpVariable.dicts("energy_discharged", range(num_hours), 0)
    battery_level = pl.LpVariable.dicts("battery_level", range(num_hours+1), 0, max_capacity)

    
    prob += pl.lpSum([energy_bought[i]*electricity_price[i] - energy_sold[i]*electricity_price[i] for i in range(num_hours)])
    # Constraints
    for i in range(num_hours):
        # Energy balance
        prob += pv_gen[i] + energy_bought[i] + energy_discharged[i]*efficiency == load[i] + energy_charged[i] + energy_sold[i] + energy_charged[i]/efficiency

        # Battery charging and discharging
        prob += energy_charged[i] <= max_capacity - battery_level[i]
        prob += energy_discharged[i] <= battery_level[i+1]

    # Battery level update
        if i < num_hours - 1:
            prob += battery_level[i+1] == battery_level[i] + efficiency*energy_charged[i] - energy_discharged[i]

    
    # Additional constraints for SoC
    prob += battery_level[0] == 0.5 * max_capacity
    prob += battery_level[24] == 0.5 * max_capacity
    
  
    return prob

pv_gen = np.empty(24,)
load = np.empty(24,)


if args.mode == 'test':
    print("Test mode: Optimal electricity bill for 08/31/2022")
    load = pd.read_csv('./load/test-083022.csv').values.flatten() # create arbitrary value
    pv_gen = pd.read_csv("pv_pred/test-083022.csv").values.flatten() # get arbitrary building
    
elif args.mode == 'eval':
    ###TO DO: combine!###
    pv_gen = pd.read_csv(PV_PATH).values.flatten()
    load = pd.read_excel(LOAD_PATH, usecols='BE', header=0).values.flatten()

else:
    raise ValueError("only 'test' or 'eval' accepted")   
    
    
prob = define_problem(prob, electricity_price=ELEC_PRICE, load=load, pv_gen=pv_gen)
# Solve the problem   
prob.solve()
    
# Print the status of the solved LP
print(f"Status: {pl.LpStatus[prob.status]}")

# Print the optimized objective function value
print(f"Optimized Cost: {round(pl.value(prob.objective))}")
    