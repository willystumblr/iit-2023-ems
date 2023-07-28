import pandas as pd
import pulp as pl
import argparse

parser = argparse.ArgumentParser(description="Scheduler Mode")
parser.add_argument('--mode', type=str, 
                    help="'test' for test with arbitary input, 'eval' for actual evaluation.")

args = parser.parse_args()

def define_problem():
    pass

if args.mode == 'test':
    pass