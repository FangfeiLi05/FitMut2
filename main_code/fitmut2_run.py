#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import fitmut2_methods

# try running with command
# python3 ./fitmut2_run.py -i ./simu_0_EvoSimulation_Read_Number.csv -t ./fitmut_input_time_points.csv -o test

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


parser.add_argument('-n', '--maximum_iteration_number', type=int, default=50,
                    help='maximum number of iterations')

parser.add_argument('-a', '--opt_algorithm', type=str, default='direct_search',
                    choices = ['direct_search','differential_evolution', 'nelder_mead'], 
                    help='choose optimization algorithm')

parser.add_argument('-p', '--parallelize', type=str, default='1',
                    help='whether to use multiprocess module to parallelize inference across lineages')

parser.add_argument('-s', '--save_steps', type=str, default='0',
                    help='whether to output files in intermediate step of iterations')
                    
parser.add_argument('-o', '--output_filename', type=str, default='output',
                    help='prefix of output .csv files')

args = parser.parse_args()


#####
read_num_seq = np.array(pd.read_csv(args.input, header=None), dtype=float)

csv_input = pd.read_csv(args.t_seq, header=None)
t_seq = np.array(csv_input[0][~pd.isnull(csv_input[0])], dtype=float)
cell_depth_seq = np.array(csv_input[1][~pd.isnull(csv_input[1])], dtype=float)

Ub = args.mutation_rate
delta_t = args.delta_t
c = args.c # per cycle
parallelize = bool(int(args.parallelize))
max_iter_num = args.maximum_iteration_number
opt_algorithm = args.opt_algorithm
output_filename = args.output_filename
save_steps = bool(int(args.save_steps))

my_obj = fitmut2_methods.FitMut(read_num_seq = read_num_seq,
                                   t_seq = t_seq,
                                   cell_depth_seq = cell_depth_seq,
                                   Ub = Ub,
                                   delta_t = delta_t,
                                   c = c,
                                   opt_algorithm = opt_algorithm,
                                   max_iter_num = max_iter_num,
                                   parallelize = parallelize,
                                   save_steps = save_steps,
                                   output_filename = output_filename)

my_obj.function_main()

