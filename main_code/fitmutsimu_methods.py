#!/usr/bin/env python3
import numpy as np
import pandas as pd
import copy
import itertools
import csv
from tqdm import tqdm

#np.random.seed(10)
np.random.seed(1)
class FitMutSimu:
    def __init__(self, lineages_num, t_pregrowth, t_seq,
                       read_num_average_seq_bundle, 
                       Ub, mutation_fitness_spectrum, max_mut_num,
                       cell_num_average_bottleneck,
                       c, dna_copies, pcr_cycles,
                       output_filename):
        
        ########################################
        self.lineages_num = lineages_num
        
        self.t_pregrowth = t_pregrowth
        self.t_seq = t_seq
        self.seq_num = len(self.t_seq)
        self.evo_num = self.t_seq[-1]
        self.delta_t = self.t_seq[1] - self.t_seq[0]
        
        self.Ub = Ub
        self.bins_edge = mutation_fitness_spectrum[0]
        self.freq_bin = mutation_fitness_spectrum[1]
        
        self.max_mut_num = max_mut_num
        
        self.cell_num_average_bottleneck = cell_num_average_bottleneck

        self.c = c

        self.dna_copies = dna_copies
        self.pcr_cycles = pcr_cycles
        
        self.read_num_average_seq_bundle = read_num_average_seq_bundle
        self.bundle_num = self.read_num_average_seq_bundle.shape[1] # double check for single (not bundle)
        
        self.output_filename = output_filename
        
        ########################################
        self.x_mean_seq = np.zeros(self.evo_num + 1, dtype=float)  # mean fitness
        
        # data at one generation (particularly, number of cell transferred for t=0, delta_t...)
        self.data_dict = {i: [1, 0, 0, 0, -self.t_pregrowth, -self.t_pregrowth] for i in range(self.lineages_num)}
        # [a0, a1, a2, a3, a4, a5]: -- a0: number of individuals
        #                           -- a1: fitness of each individual (sum of fitness of all mutations)
        #                           -- a2: number of mutations in each individual
        #                           -- a3: fitness of the newest mutation (0 for the neutral)
        #                           -- a4: occurance time of the newest mutation
        #                           -- a5: establishment time of the newest mutation
         
    
          
    ##################################################
    def function_mutation_fitness_spectrum_random_generator(self, n):
        """
        ------------------------------------------------------------
        A SUB-FUNCTION CALLED BY FUNCTION 
        function_update_data_growth_lineage() TO GENERATE RANDOM 
        NUMBERS THAT FOLLOW AN ARRBITRARY PROBABILITY DISTRIBUTION
    
        INPUTS
        - n: number of random numbers generated
            
        OUTPUTS
        - output: a vector of n random numbers
        
        SELF
        - bins_edge: the bin edge vector, [x_0, x_1, x_2, ..., x_n]
        - freq_bin: the corresponding count frequency in each bin, 
                    [f_0, f_1, ..., f_{n-1}] (e.g., f_0 for [x_0, x_1))
        ------------------------------------------------------------
        """        
        freq_bin_accum = np.cumsum(self.freq_bin)
        
        output = []
        j = 0
        while j < int(n):
            tmp = np.random.rand(1)
            pos = np.where(tmp < freq_bin_accum)[0][0]
            l1, l2 = self.bins_edge[pos], self.bins_edge[pos + 1]
            output.append(l1 + (l2 - l1) * tmp[0])
            j += 1
       
        return output
        
        
        
    ##################################################
    def function_growth_lineage(self, data_lineage_dict): #growth
        """ """
        if (not isinstance(data_lineage_dict, dict) or not data_lineage_dict): #final layer
            if data_lineage_dict[0] > 0:
                data_lineage_dict[0] = 2 * data_lineage_dict[0] * (1 + data_lineage_dict[1])
            return data_lineage_dict
    
        else:
            for (item_key, item_value) in data_lineage_dict.items():
                data_lineage_dict[item_key] = self.function_growth_lineage(item_value)
            return data_lineage_dict
            
    
    ##################################################
    def function_mutation_lineage(self, data_lineage_dict, t): #mutation
        """ """
        if (not isinstance(data_lineage_dict, dict) or not data_lineage_dict): #final layer
            if (data_lineage_dict[0] > 0) and (data_lineage_dict[2] < self.max_mut_num):
                tmp = np.copy(data_lineage_dict)
                mut_num = np.random.binomial(tmp[0], self.Ub)
                if mut_num > 0:
                    data_lineage_dict = dict()
                    data_lineage_dict[0] = tmp
                    tmp_random = self.function_mutation_fitness_spectrum_random_generator(mut_num)
                    for i1 in range(mut_num):
                        data_lineage_dict[i1+1] = [1, tmp[1] + tmp_random[i1], tmp[2] + 1, tmp_random[i1], t, -self.t_pregrowth]
            return data_lineage_dict
    
        else:
            for (item_key, item_value) in data_lineage_dict.items():
                data_lineage_dict[item_key] = self.function_mutation_lineage(item_value, t)
            return data_lineage_dict
    
    
    ##################################################
    def function_sampling_lineage(self, data_lineage_dict): #sampling
        """ """
        if (not isinstance(data_lineage_dict, dict) or not data_lineage_dict): #final layer
            if data_lineage_dict[0] > 0:
                data_lineage_dict[0] = np.random.poisson(data_lineage_dict[0] * self.ratio)
            return data_lineage_dict
    
        else:
            for (item_key, item_value) in data_lineage_dict.items():
                data_lineage_dict[item_key] = self.function_sampling_lineage(item_value)
            return data_lineage_dict
            
            
    ##################################################
    def function_amplifying_lineage(self, data_lineage_dict): #amplifying
        """ """
        if (not isinstance(data_lineage_dict, dict) or not data_lineage_dict): #final layer
            if data_lineage_dict[0] > 0:
                for r in range(self.pcr_cycles):
                    data_lineage_dict[0] = np.random.poisson(2 * data_lineage_dict[0])
            return data_lineage_dict
    
        else:
            for (item_key, item_value) in data_lineage_dict.items():
                data_lineage_dict[item_key] = self.function_amplifying_lineage(item_value)
            return data_lineage_dict
    
    
    ##################################################
    def function_update_data_population(self, data_population_dict, t, phase_choice):
        """ """
        for i in range(self.lineages_num):
            if phase_choice == 'growth':
                data_population_dict[i] = self.function_growth_lineage(data_population_dict[i])
            
            elif phase_choice == 'mutation':
                data_population_dict[i] = self.function_mutation_lineage(data_population_dict[i], t)
            
            elif phase_choice == 'sampling':
                data_population_dict[i] = self.function_sampling_lineage(data_population_dict[i])
                
            elif phase_choice == 'amplifying':
                data_population_dict[i] = self.function_amplifying_lineage(data_population_dict[i])
        
        return data_population_dict
        
        
        
    ################################################## might need to change
    def function_check_establishment_lineage(self, data_lineage_dict, t, phase_choice): # uncompeted_growth, competed_growth
        """
        """
        if (not isinstance(data_lineage_dict, dict) or not data_lineage_dict): # final layer
            if data_lineage_dict[0] > 0:
                if (data_lineage_dict[3]>0) and (data_lineage_dict[5]==-self.t_pregrowth): # if there exist mutation, and it hasn't established yet
                    if phase_choice == 'uncompeted_growth':
                        establishment_size = 2*self.c/np.max([data_lineage_dict[1], 0.005])  # double check: fitness of individual
                        if data_lineage_dict[0]*self.ratio >= establishment_size:
                            data_lineage_dict[5] = t
                        
                    elif phase_choice == 'competed_growth':
                        fitness_relative = np.max([(1 + data_lineage_dict[1]) / (1 + self.x_mean_seq[t-1]), 0]) # double check: fitness of individual
                        establishment_size = 2*self.c/np.max([fitness_relative-1, 0.005])  
                        if data_lineage_dict[0]*self.ratio >= establishment_size:
                            data_lineage_dict[5] = t
                 
            return data_lineage_dict
        
        else:
            for (item_key, item_value) in data_lineage_dict.items():
                data_lineage_dict[item_key] = self.function_check_establishment_lineage(item_value, t, phase_choice)
                #% assign counter-th value to this leaf
                #counter = counter+1
            
            return data_lineage_dict
            
            
    ##################################################
    def function_check_establishment_population(self, data_population_dict, t, phase_choice): # uncompeted_growth, competed_growth
        """ """
        for i in range(self.lineages_num):
            data_population_dict[i] = self.function_check_establishment_lineage(data_population_dict[i], t, phase_choice)
          
        return data_population_dict
    
    

    
        
    ##################################################
    def function_extract_data_info_lineage(self, data_lineage_dict):
        """
        Extract mutation information from a single lineage
        """
        if (not isinstance(data_lineage_dict, dict) or not data_lineage_dict):
            self.data_info_lineage['cell_number'].append(data_lineage_dict[0])
            self.data_info_lineage['cell_fitness'].append(data_lineage_dict[1])
            self.data_info_lineage['mutation_number'].append(data_lineage_dict[2])
            self.data_info_lineage['mutation_fitness'].append(data_lineage_dict[3])
            self.data_info_lineage['occurance_time'].append(data_lineage_dict[4])
            self.data_info_lineage['establishment_time'].append(data_lineage_dict[5])
        else:
            for (item_key, item_value) in data_lineage_dict.items():
                self.function_extract_data_info_lineage(item_value)

    
    
    ##################################################
    def function_calculate_mean_fitness(self, t):
        """
        """
        tmp_numerator = 0    # numerator/denominator
        tmp_denominator = 0
        
        for i in range(self.lineages_num):
            self.data_info_lineage = {'mutation_fitness': [], 
                                      'occurance_time':[], 
                                      'establishment_time':[], 
                                      'mutation_number':[], 
                                      'cell_fitness':[], 
                                      'cell_number':[]}
                                                
            self.function_extract_data_info_lineage(self.data_dict[i])
        
            tmp_numerator += np.dot(self.data_info_lineage['cell_fitness'], self.data_info_lineage['cell_number'])
            tmp_denominator += np.sum(self.data_info_lineage['cell_number'])
         
        self.x_mean_seq[t] = tmp_numerator/tmp_denominator
        
    
    
    ##################################################
    def function_extract_data_info(self, data_dict):
        """
        Extract the cell number of all lineages
        """
        data_info = {'all': np.zeros(self.lineages_num, dtype=float), 
                     'neutral': np.zeros(self.lineages_num, dtype=float)}
        
        for i in range(self.lineages_num):            
            self.data_info_lineage = {'mutation_fitness': [], 
                                      'occurance_time':[], 
                                      'establishment_time':[], 
                                      'mutation_number':[], 
                                      'cell_fitness':[],
                                      'cell_number':[]}

            self.function_extract_data_info_lineage(data_dict[i])
            
            data_info['all'][i] = np.sum(self.data_info_lineage['cell_number'])
            
            for m in list(set(self.data_info_lineage['mutation_number'])):
                tmp = np.dot(np.array(self.data_info_lineage['mutation_number']) == m, 
                             np.array(self.data_info_lineage['cell_number']))
                
                if m == 0:
                     data_info['neutral'][i] = tmp
                else:
                    if 'mutation_' + str(m) not in list(data_info.keys()):
                        data_info['mutation_' + str(m)] = np.zeros(self.lineages_num, dtype=float)
                    data_info['mutation_' + str(m)][i] = tmp

        return data_info
    

    

    ##################################################
    def funciton_save_data(self):
        """ """
        ####################
        data_mut_info = {'mutation_fitness':[], 
                         'occurance_time':[], 
                         'establishment_time':[], 
                         'mutation_number':[],
                         'cell_number':[],
                         'lineage_label':[]} # not finish
        
        for i in range(self.lineages_num):
            self.data_info_lineage = {'mutation_fitness': [], 
                                      'occurance_time':[], 
                                      'establishment_time':[], 
                                      'mutation_number':[], 
                                      'cell_fitness':[],
                                      'cell_number':[]}
            
            self.function_extract_data_info_lineage(self.data_dict[i])
            
            #idx = np.where(np.array(self.data_info_lineage['mutation_fitness'])>0)[0] # save all mutaitons
            idx = np.where(np.array(self.data_info_lineage['establishment_time']) > -self.t_pregrowth)[0]
            if len(idx)>0:
                for i_idx in idx:                
                    data_mut_info['mutation_fitness'].append(np.array(self.data_info_lineage['mutation_fitness'])[i_idx])
                    data_mut_info['occurance_time'].append(np.array(self.data_info_lineage['occurance_time'])[i_idx])
                    data_mut_info['establishment_time'].append(np.array(self.data_info_lineage['establishment_time'])[i_idx])
                    data_mut_info['mutation_number'].append(np.array(self.data_info_lineage['mutation_number'])[i_idx])
                    data_mut_info['cell_number'].append(np.array(self.data_info_lineage['cell_number'])[i_idx])
                    data_mut_info['lineage_label'].append(i)
                
        
        tmp = list(itertools.zip_longest(*list(data_mut_info.values())))
        with open(self.output_filename + '_EvoSimulation_Mutation_Info.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(data_mut_info.keys())
            w.writerows(tmp)
        
        
        self.cell_num_mutant_fraction_seq = np.zeros(self.seq_num, dtype=float)
        for k in range(self.seq_num):
            self.cell_num_mutant_fraction_seq[k] = ((np.sum(self.cell_num_seq_dict['all'][:,k]) 
                                                     - np.sum(self.cell_num_seq_dict['neutral'][:,k])) 
                                                    / np.sum(self.cell_num_seq_dict['all'][:,k]))

        ####################
        data_other = {'Time_Points': self.t_seq, 
                      'Mean_Fitness': self.x_mean_seq[self.t_seq],
                      'lineages_Number': [self.lineages_num],
                      'gDNA_Copies': [self.dna_copies], 
                      'PCR_cycles': [self.pcr_cycles], 
                      'Ub': [self.Ub], 
                      'Mutant_Cell_Fraction': self.cell_num_mutant_fraction_seq, 
                      'Average_Read_Depth': self.read_num_average_seq_bundle}
        
        tmp = list(itertools.zip_longest(*list(data_other.values())))
        with open(self.output_filename + '_EvoSimulation_Other_Info.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(data_other.keys())
            w.writerows(tmp)
        
        
        tmp = pd.DataFrame(self.cell_num_seq_dict['all'])
        tmp.to_csv(self.output_filename + '_EvoSimulation_Saturated_Cell_Number.csv',
                     index=False, header=False)
                     
        tmp = pd.DataFrame(self.cell_num_seq_dict['neutral'])
        tmp.to_csv(self.output_filename + '_EvoSimulation_Saturated_Cell_Number_Neutral.csv',
                     index=False, header=False)
                     

        tmp = pd.DataFrame(self.bottleneck_cell_num_seq_dict['all'])
        tmp.to_csv(self.output_filename + '_EvoSimulation_Bottleneck_Cell_Number.csv',
                     index=False, header=False)
                     
        tmp = pd.DataFrame(self.bottleneck_cell_num_seq_dict['neutral'])
        tmp.to_csv(self.output_filename + '_EvoSimulation_Bottleneck_Cell_Number_Neutral.csv',
                     index=False, header=False)


        
        
        ####################
        for bundle_idx in range(self.bundle_num):
            tmp = pd.DataFrame(self.read_num_seq_bundle_dict[bundle_idx]['all'])
            tmp.to_csv(self.output_filename + '_' + str(bundle_idx) + '_EvoSimulation_Read_Number.csv',
                         index=False, header=False)

        

    ##################################################
    def function_main(self):
        """
        """
        # Step 1: pre-evolution (to simulate the process of building barcode library)
        for t in tqdm(range(-self.t_pregrowth+1, 1)):  # -15, -14, -13, ..., -1, 0 (16 generations in total)
            self.data_dict = self.function_update_data_population(self.data_dict, t, 'growth') # -- deterministic growth
            
            self.ratio = 1
            self.data_dict = self.function_update_data_population(self.data_dict, t, 'sampling') # -- sampling
            
            self.data_dict = self.function_update_data_population(self.data_dict, t, 'mutation') # -- mutation
                       
            # -- check establishment
            self.ratio = 1 / 2 ** (self.t_pregrowth+t)
            self.data_dict = self.function_check_establishment_population(self.data_dict, t, 'uncompeted_growth')

                
        
        # -- lineage filter: remove lineages with 0 cell number at t=0
        output = self.function_extract_data_info(self.data_dict)
        idx_keep = np.where(output['all'] > 0)[0]
        self.lineages_num = len(idx_keep)
        self.data_dict_copy = {i: copy.deepcopy(self.data_dict[idx_keep[i]]) for i in range(self.lineages_num)}
        self.data_dict = self.data_dict_copy
        
        
        # Step 2: evolution (to simulate the whole process of the evolution)
        self.cell_num_seq_dict = dict()
        self.bottleneck_cell_num_seq_dict = dict()
        self.read_num_seq_bundle_dict = {i: dict() for i in range(self.bundle_num)}

        for t in tqdm(range(self.evo_num + 1)):
            if t == 0: #Step 2-1: initialization (to simulate the process of evolution initialization)
                depth = np.sum(self.function_extract_data_info(self.data_dict)['all'])
                self.ratio = 2 ** self.delta_t * self.cell_num_average_bottleneck * self.lineages_num / depth
                self.data_dict = self.function_update_data_population(self.data_dict, t, 'sampling')
                self.data_saturated_dict = copy.deepcopy(self.data_dict)
                
                self.ratio = 1 / 2 ** self.delta_t
                self.data_dict = self.function_update_data_population(self.data_dict, t, 'sampling')
            
            
            else: # Step 2-2: cell growth (to simulate growth of of one generation)
                self.data_dict = self.function_update_data_population(self.data_dict, t, 'growth') # -- growth part1: simulate deterministic growth

                depth = np.sum(self.function_extract_data_info(self.data_dict)['all'])
                self.ratio = 2 ** (np.mod(t-1, self.delta_t) + 1) * self.cell_num_average_bottleneck * self.lineages_num / depth
                self.data_dict = self.function_update_data_population(self.data_dict, t, 'sampling') # -- growth part2: add growth noise
                
                self.data_dict = self.function_update_data_population(self.data_dict, t, 'mutation') # -- growth part3: add mutations
                
                # -- check establishment
                depth = np.sum(self.function_extract_data_info(self.data_dict)['all'])
                self.ratio = self.cell_num_average_bottleneck * self.lineages_num * self.delta_t / depth # numerator is effective cell number
                #self.ratio = self.cell_num_average_bottleneck * self.lineages_num / depth # numerator is effective cell number
                self.data_dict = self.function_check_establishment_population(self.data_dict, t, 'competed_growth')
                            
     
                # Step 2-3: cell transfer (to simulate sampling of cell transfer at the bottleneck)
                mode_factor = np.mod(t, self.delta_t)
                if mode_factor == 0: # bottlenecks
                    self.data_saturated_dict = copy.deepcopy(self.data_dict)
                    self.ratio = 1 / 2 ** self.delta_t
                    self.data_dict = self.function_update_data_population(self.data_dict, t, 'sampling') # -- sampling
                                    
            
            # -- calculate the mean_fitness at the generation t
            self.function_calculate_mean_fitness(t)
                
            
            mode_factor = np.mod(t, self.delta_t)
            if mode_factor == 0:
                k = int(t/self.delta_t)

                # -- save data of samples at bottleneck
                output_bottleneck = self.function_extract_data_info(self.data_dict)
                for key in output_bottleneck.keys():
                    if key not in self.bottleneck_cell_num_seq_dict.keys():
                        self.bottleneck_cell_num_seq_dict[key] = np.zeros((self.lineages_num, self.seq_num), dtype=int)
                    self.bottleneck_cell_num_seq_dict[key][:,k] = output_bottleneck[key]
                
                
                # -- save data of samples at saturated (the samples are used for the following DNA extraction and PCR)
                output_saturated = self.function_extract_data_info(self.data_saturated_dict)
                for key in output_saturated.keys():
                    if key not in self.cell_num_seq_dict.keys():
                        self.cell_num_seq_dict[key] = np.zeros((self.lineages_num, self.seq_num), dtype=int)
                    self.cell_num_seq_dict[key][:,k] = output_saturated[key]

     
                # Step 2-4: DNA extraction (to simulate the processes of extracting genome DNA)
                depth = np.sum(output_saturated['all'])
                self.ratio = self.dna_copies * self.lineages_num / depth
                self.data_dna_dict = copy.deepcopy(self.data_saturated_dict)
                self.data_dna_dict = self.function_update_data_population(self.data_dna_dict, t, 'sampling') # -- sampling
                
                # Step 2-5: PCR (to simulate the processes of running PCR)
                self.data_pcr_dict = copy.deepcopy(self.data_dna_dict)
                self.data_pcr_dict = self.function_update_data_population(self.data_pcr_dict, t, 'amplifying')
            
                # Step 2-6: sequencing (to simulate the processes of sequencing)
                depth = np.sum(self.function_extract_data_info(self.data_pcr_dict)['all'])
                for bundle_idx in range(self.bundle_num):
                    self.read_num_average_seq = self.read_num_average_seq_bundle[:,bundle_idx]  # might need to change
                    self.ratio = self.read_num_average_seq[k] * self.lineages_num / depth
                    self.data_sequencing_dict = copy.deepcopy(self.data_pcr_dict)
                    self.data_sequencing_dict = self.function_update_data_population(self.data_sequencing_dict, t, 'sampling')

                        
                    # -- save read number data for sequencing results
                    output = self.function_extract_data_info(self.data_sequencing_dict)
                    for idx in output.keys():
                        if idx not in self.read_num_seq_bundle_dict[bundle_idx].keys():
                            self.read_num_seq_bundle_dict[bundle_idx][idx] = np.zeros((self.lineages_num, self.seq_num), dtype=int)
                        self.read_num_seq_bundle_dict[bundle_idx][idx][:,k] = output[idx]
        
        
        # Step: save data
        self.funciton_save_data() 
        