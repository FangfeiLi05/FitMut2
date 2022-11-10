#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import fitmutsimu_methods

# try running with command 
#python fitmutsimu_methods.py -l 100000 -t simu_input_time_points.csv -s simu_input_mutation_fitness.csv -o test


###########################
##### PARSE ARGUMENTS #####
###########################

parser = argparse.ArgumentParser(description = 'Simulated evolution of a isogentic population', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
        
parser.add_argument('-l', '--lineage_number', type=int, required=True,
                    help = 'number of lineages started as transformation')

parser.add_argument('-t', '--t_seq', type=str, required=True,
                    help = 'a .csv file, with:'
                           '1st column: sequenced time-points evaluated in number of generations, '
                           '2nd+ column: average number of reads per barcode for each sequenced time-point (accept multiple columns for multiple sequencing runs).')

parser.add_argument('-s', '--mutation_fitness', type=str, required='', 
                    help = 'a .csv file, with:'
                           '1st column: total beneficial mutation rate, Ub, '
                           '2nd column: bin edges of the arbitrary DFE, '
                           '3rd column: counts frequency in each bin of the 2nd column.')

parser.add_argument('-max_mut_num', '--maximum_mutation_number', type=int, default=1,
                    help = 'maximum number of mutations allowed in each single cell')

parser.add_argument('-t_pre', '--t_pregrowth', type=int, default=16,
                    help = 'number of genrations in pre-growth (prior to the evolution)')

parser.add_argument('-n_b', '--cell_num_average_bottleneck', type=int, default=100,
                    help = 'average number of cells per barcode transferred at each bottleneck')

parser.add_argument('-c', '--c', type=float, default=1, 
                    help='2c = half of variance introduced by cell growth and cell transfer')
    
parser.add_argument('-d', '--dna_copies', type=int, default=500, 
                    help = 'average genome copy number per barcode used as template in PCR')

parser.add_argument('-p', '--pcr_cycles', type=int, default=25, 
                    help = 'number of cycles in PCR')

parser.add_argument('-o', '--output_filename', type=str, default='output', 
                    help = 'prefix of output .csv files')

args = parser.parse_args()


##########
lineages_num = args.lineage_number
t_pregrowth = args.t_pregrowth
    
csv_input = pd.read_csv(args.t_seq, header=None)
t_seq = np.array(csv_input[0][~pd.isnull(csv_input[0])], dtype=int)
read_num_average_seq_bundle = np.array(csv_input.loc[:,1:csv_input.shape[1]], dtype=int)

csv_input = pd.read_csv(args.mutation_fitness, header=None)
Ub = np.array(csv_input[0][~pd.isnull(csv_input[0])], dtype=float)[0]
mutation_fitness_spectrum = [np.array(csv_input[1][~pd.isnull(csv_input[1])], dtype=float), 
                    np.array(csv_input[2][~pd.isnull(csv_input[2])], dtype=float)]

max_mut_num = args.maximum_mutation_number 

cell_num_average_bottleneck = args.cell_num_average_bottleneck

c = args.c
dna_copies = args.dna_copies
pcr_cycles = args.pcr_cycles
output_filename = args.output_filename

my_obj = fitmutsimu_methods.FitMutSimu(lineages_num = lineages_num,
                                          t_pregrowth = t_pregrowth,
                                          t_seq = t_seq,
                                          read_num_average_seq_bundle = read_num_average_seq_bundle,
                                          Ub = Ub,
                                          mutation_fitness_spectrum = mutation_fitness_spectrum,
                                          max_mut_num = max_mut_num,
                                          cell_num_average_bottleneck = cell_num_average_bottleneck,
                                          c = c,
                                          dna_copies = dna_copies,
                                          pcr_cycles = pcr_cycles,
                                          output_filename = output_filename)

my_obj.function_main()

