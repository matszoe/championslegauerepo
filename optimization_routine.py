import numpy as np
import pandas as pd
import pyomo as pyo

def load_input_data(input_file_path):
    
    return input_data

def setup_optimization_problem():
    
    model = pyo.ConcreteModel()

    ## Define Variables

    ## Define Constraints

    ## Define Objective Function


    return model


def optimize(model):
    solver = pyo.SolverFactory('gurobi')
    results = solver.solve(model)

    return results

def process_results(results):
    
    pass