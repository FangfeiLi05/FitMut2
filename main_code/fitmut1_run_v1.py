#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import fitmut1_methods_v1

# try running with command
# python3 ./fitmut1_run.py -i ./simu_0_EvoSimulation_Read_Number.csv -t ./fitmut_input_time_points.csv -o test

###########################
##### PARSE ARGUMENTS #####
###########################
parser = argparse.ArgumentParser(description='Estimate fitness and establishment time of spontaneous adaptive mutations in a competitive pooled growth experiment', 
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
parser.add_argument('-i', '--input', type=str, required=True,
                     help='a .csv file: with each column being the read number per barcode at each sequenced time-point')

parser.add_argument('-t', '--t_seq', type=str, required=True,
                    help='a .csv file of 2 columns:'
                         '1st column: sequenced time-points evaluated in number of generations, '
                         '2nd column: total effective number of cells of the population for each sequenced time-point.')
                    
parser.add_argument('-u', '--mutation_rate', type=float, default=1e-5, 
                    help='total beneficial mutation rate')

parser.add_argument('-dt', '--delta_t', type=float, default=8, 
                    help='number of generations between bottlenecks')

parser.add_argument('-c', '--c', type=float, default=1, 
                    help='half of variance introduced by cell growth and cell transfer')

parser.add_argument('-l1', '--l1', type=int, default=20, 
                    help='neutral_limitation_left')

parser.add_argument('-l2', '--l2', type=int, default=30, 
                    help='neutral_limitation_right')

parser.add_argument('-a', '--opt_algorithm', type=str, default='nelder_mead',
                    choices = ['differential_evolution', 'nelder_mead'], 
                    help='choose optmization algorithm')

parser.add_argument('-p', '--parallelize', type=bool, default=True,
                    help='whether to use multiprocess module to parallelize inference across lineages')
 
parser.add_argument('-o', '--output_filename', type=str, default='output',
                    help='prefix of output .csv files')

args = parser.parse_args()


##################################################
read_num_seq = np.array(pd.read_csv(args.input, header=None), dtype=float)

csv_input = pd.read_csv(args.t_seq, header=None)
t_seq = np.array(csv_input[0][~pd.isnull(csv_input[0])], dtype=float)
cell_depth_seq = np.array(csv_input[1][~pd.isnull(csv_input[1])], dtype=float)

Ub = args.mutation_rate
delta_t = args.delta_t
c = args.c
l1 = args.l1
l2 = args.l2
parallelize = args.parallelize
opt_algorithm = args.opt_algorithm
output_filename = args.output_filename

my_obj = fitmut1_methods_v1.FitMut(read_num_seq = read_num_seq,
                                   t_seq = t_seq,
                                   cell_depth_seq = cell_depth_seq,
                                   Ub = Ub,
                                   delta_t = delta_t,
                                   c = c,
                                   l1 = l1,
                                   l2 = l2,
                                   opt_algorithm = opt_algorithm,
                                   parallelize = parallelize,
                                   output_filename = output_filename)

my_obj.function_main()


