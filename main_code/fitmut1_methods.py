#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import Bounds
from scipy.optimize import differential_evolution
import math
import itertools
import csv
import time
from multiprocess import Pool, Process
from tqdm import tqdm


# fitness inference object
class FitMut:
    def __init__(self, read_num_seq, 
                       t_seq, 
                       cell_depth_seq, 
                       Ub, 
                       delta_t,
                       c,
                       l1, 
                       l2, 
                       opt_algorithm, 
                       parallelize,
                       output_filename):
        
        # preparing inputs (meaning of some can be found in fitmut_new_run.py)
        self.Ub = Ub
        self.delta_t = delta_t
        #self.noise_c = c / delta_t # self.noise_c: noise per generation, c: noise per cycle
        self.noise_c = c # self.noise_c: noise per cycle

        self.read_num_t1_left = l1
        self.read_num_t1_right = l2
        self.opt_algorithm = opt_algorithm
        self.parallelize = parallelize
        self.output_filename = output_filename

        self.read_num_seq = read_num_seq
        self.read_depth_seq = np.sum(self.read_num_seq, axis=0)
        self.lineages_num, self.seq_num = np.shape(self.read_num_seq) 
        self.t_seq = t_seq
        self.cell_depth_seq = cell_depth_seq
        self.ratio = np.true_divide(self.read_depth_seq, self.cell_depth_seq)
        self.cell_num_seq = self.read_num_seq / self.read_depth_seq * self.cell_depth_seq
        self.cell_num_seq[self.cell_num_seq < 1] = 1 
        self.read_num_seq[self.read_num_seq < 1] = 1 
        
        self.delta_t = 8

        # set bounds for the optimization
        self.bounds_0 = Bounds([1e-5, .5], [.5, 6])

        if self.opt_algorithm == 'differential_evolution':
            self.bounds_1 = Bounds([1e-8, -100], [.5, math.floor(self.t_seq[-1] - 1)])
        elif self.opt_algorithm == 'nelder_mead':
            self.bounds_1 = [[1e-8, .5], [-100, math.floor(self.t_seq[-1] - 1)]]


    #####
    def function_mean_fitness_kappa_opt(self, x):
        """
        Optimization function for calculating mean fitness and kappa value for each time-point
        """
        x_mean = x[0]
        kappa = x[1]
        
        read_num_t2_est = (self.read_num_t1 * np.exp(-x_mean*(self.t_seq_t1t2[1] - self.t_seq_t1t2[0])) 
                            * self.ratio_t1t2[1] / self.ratio_t1t2[0])
        
        read_num_t2 = np.arange(self.read_num_t2_left, self.read_num_t2_right)
        pdf_conditional_theory = np.multiply(np.sqrt(np.sqrt(read_num_t2_est)
                                                    / (4 * np.pi * kappa * np.power(read_num_t2, 1.5))),
                                            np.exp(-np.power(np.sqrt(read_num_t2_est)
                                                             - np.sqrt(read_num_t2), 2) / kappa))#*self.num

        square_sum_pdf = np.sum((pdf_conditional_theory - self.pdf_conditional_measure)**2)
        
        return square_sum_pdf
    
    
    #####
    def function_mean_fitness_kappa(self):
        """
        Calculate mean fitness and kappa value for each time-point using optimization
        """        
        self.kappa_seq = 2 * np.ones(self.seq_num, dtype=float)
        self.x_mean_seq = 1e-5 * np.ones(self.seq_num, dtype=float)
        
        for k in range(self.seq_num-1):
            # look at initial read numbers between self.read_num_t1_left and self.read_num_t1_right
            x_mean_tmp = np.zeros(self.read_num_t1_right - self.read_num_t1_left, dtype=float)
            kappa_tmp = np.zeros(self.read_num_t1_right - self.read_num_t1_left, dtype=float)
            
            self.ratio_t1t2 = self.ratio[k:k+2]
            self.t_seq_t1t2 = self.t_seq[k:k+2]
            
            for self.read_num_t1 in range(self.read_num_t1_left, self.read_num_t1_right):
                #pos = self.read_num_seq[:, k] == self.read_num_t1
                pos = np.abs(self.read_num_seq[:, k] - self.read_num_t1)<=1
                self.num = np.sum(pos)
                
                tmp = 10
                self.read_num_t2_left = np.max([np.min(self.read_num_seq[pos, k+1]), tmp])
                self.read_num_t2_right = np.max(self.read_num_seq[pos, k+1])
                
                self.pdf_conditional_measure = np.histogram(self.read_num_seq[pos, k+1], 
                                                           bins=np.arange(self.read_num_t2_left, 
                                                                          self.read_num_t2_right+2), 
                                                           density=True)[0][:-1]
                self.pdf_conditional_measure = self.pdf_conditional_measure * np.sum(self.read_num_seq[pos, k+1]>=tmp)/self.num
                
                #self.pdf_conditional_measure = np.histogram(self.read_num_seq[pos, k+1], 
                #                                           bins=np.arange(self.read_num_t2_left, 
                #                                                          self.read_num_t2_right+2), 
                #                                           density=False)[0][:-1]

                #opt_output = minimize(fun = self.function_mean_fitness_kappa_opt,
                #                      x0 = [5e-8, 2.5],
                #                      method = 'BFGS',
                #                      options = {'disp':False, 'maxiter':1000}) #'gtol': 1e-10, 'eps': 1e-3
                
                opt_output = differential_evolution(func = self.function_mean_fitness_kappa_opt,
                                                    seed = 1, 
                                                    bounds = self.bounds_0)

                x_mean_tmp[self.read_num_t1 - self.read_num_t1_left] = opt_output.x[0]
                kappa_tmp[self.read_num_t1 - self.read_num_t1_left] = opt_output.x[1]
            
            self.x_mean_seq[k+1] = np.max([np.mean(x_mean_tmp), 1e-5])
            self.kappa_seq[k+1] = np.mean(kappa_tmp)
    
    
    #####
    def function_sum_term(self):
        """
        Calculate sum_term
        """
        self.t_seq_extend = np.concatenate((-np.arange(self.delta_t, 100+self.delta_t, self.delta_t)[::-1], self.t_seq))
        self.x_mean_seq_extend = np.concatenate((1e-5*np.ones(len(self.t_seq_extend) - self.seq_num), self.x_mean_seq))

        seq_num_extend = len(self.t_seq_extend)
        self.sum_term = np.ones(seq_num_extend)
        
        sum_term_tmp = 0
        for k in range(1, seq_num_extend):
            sum_term_tmp += (self.t_seq_extend[k] - self.t_seq_extend[k-1]) * self.x_mean_seq_extend[k]        
            self.sum_term[k] = np.exp(-sum_term_tmp)



    #####
    def function_establishment_size(self, s, tau):
        """
        Calculate the establishment size for a mutation
        """
        x_mean_tau = np.interp(tau, self.t_seq_extend, self.x_mean_seq_extend)
        
        established_size = self.noise_c / np.max([s-x_mean_tau, 0.005])
        
        return established_size
    
    
    #####
    def function_cell_num_theory_lineage(self, s, tau):
        """
        Calculate estimated cell number & mutant cell number for a lineage for each time-point
        """ 
        seq_num = len(self.t_seq)
        cell_num_seq_lineage_theory = np.zeros(seq_num, dtype=float)
        cell_num_seq_lineage_theory[0] = self.cell_num_seq_lineage[0]

        mutant_cell_num_seq_lineage_theory = np.zeros(seq_num, dtype=float)
        unmutant_cell_num_seq_lineage_theory = np.zeros(seq_num, dtype=float)
    
        established_size = self.function_establishment_size(s, tau)
        tmp_tau = np.interp(tau, self.t_seq_extend, self.sum_term)

        for k in range(1, seq_num+1):
            tmp_kminus1 = np.interp(self.t_seq[k-1], self.t_seq_extend, self.sum_term)
            tmp = established_size * np.exp(s * (self.t_seq[k-1] - tau)) * tmp_kminus1 / tmp_tau
                
            mutant_cell_num_seq_lineage_theory[k-1] = np.min([tmp, self.cell_num_seq_lineage[k-1]])
            unmutant_cell_num_seq_lineage_theory[k-1] = self.cell_num_seq_lineage[k-1] - mutant_cell_num_seq_lineage_theory[k-1]
            
            if k < seq_num:
                tmp_k = np.interp(self.t_seq[k], self.t_seq_extend, self.sum_term)
                cell_num_seq_lineage_theory[k] = (unmutant_cell_num_seq_lineage_theory[k-1] + mutant_cell_num_seq_lineage_theory[k-1] * np.exp(s * (self.t_seq[k] - self.t_seq[k-1]))) * tmp_k / tmp_kminus1
        

        return cell_num_seq_lineage_theory, mutant_cell_num_seq_lineage_theory

    
    #####
    def function_noise_distribution(self, read_num_seq_lineage_theory):
        """
        Calculate the density of noise distribution (in log)
        """
        density_value_log = (0.25 * np.log(read_num_seq_lineage_theory) - 0.5 * np.log(4 * np.pi * self.kappa_seq)
                             - 0.75 * np.log(self.read_num_seq_lineage)
                             - (np.sqrt(self.read_num_seq_lineage) - np.sqrt(read_num_seq_lineage_theory)) ** 2 / self.kappa_seq)
        
        #density_value = ((read_num_seq_lineage_theory)**(0.25)
        #                 / (4 * np.pi * self.kappa_seq)**(0.5)
        #                 / self.read_num_seq_lineage**(0.75)
        #                 *np.exp(-(np.sqrt(self.read_num_seq_lineage)
        #                           - np.sqrt(read_num_seq_lineage_theory)) ** 2 / self.kappa_seq))
    
        #density_value = (np.exp(-(read_num_seq_lineage_theory + self.read_num_seq_lineage)/self.kappa_seq)
        #                 * (1/self.kappa_seq) * np.sqrt(read_num_seq_lineage_theory/self.read_num_seq_lineage)
        #                 * special.iv(1, 2*np.sqrt(self.read_num_seq_lineage * read_num_seq_lineage_theory)/self.kappa_seq))
        #
        #density_value_log = np.log(density_value)
    
        return density_value_log
    

    ##### 
    def function_likelihood_lineage_adaptive(self, x):
        """
        Calculate negative log likelihood value of an adaptive lineage
        """
        s, tau = x[0], x[1]
    
        cell_num_seq_lineage_theory_adaptive, _ = self.function_cell_num_theory_lineage(s, tau)
        read_num_seq_lineage_theory_adaptive = np.multiply(cell_num_seq_lineage_theory_adaptive, self.ratio)
        read_num_seq_lineage_theory_adaptive[read_num_seq_lineage_theory_adaptive<1] = 1

        log_likelihood_seq_lineage_adaptive = self.function_noise_distribution(read_num_seq_lineage_theory_adaptive)
      
        delta_s = 0.005
        tmp1 = self.Ub * np.exp(-s / 0.1) / 0.1 * delta_s
        
        probability_prior_adaptive_log = np.log(tmp1* np.max([s, 1e-8]) * self.cell_num_seq_lineage[0])

        log_likelihood_lineage = np.sum(log_likelihood_seq_lineage_adaptive) + probability_prior_adaptive_log
    
        return -log_likelihood_lineage

    
    #####
    def function_likelihood_lineage_neutral(self, x):
        """
        Calculate negative log likelihood value of a neutral lineage
        -- doesn't depend on s and tau, might removed???
        """
        s, tau = x[0], x[1]
    
        cell_num_seq_lineage_theory_neutral, _ = self.function_cell_num_theory_lineage(s, tau)
        read_num_seq_lineage_theory_neutral = np.multiply(cell_num_seq_lineage_theory_neutral, self.ratio)
        read_num_seq_lineage_theory_neutral[read_num_seq_lineage_theory_neutral<1] = 1
    
        log_likelihood_seq_lineage_neutral = self.function_noise_distribution(read_num_seq_lineage_theory_neutral)
        
        log_likelihood_lineage = np.sum(log_likelihood_seq_lineage_neutral)
    
        return log_likelihood_lineage
    
  
    #####
    def function_bound_points(self, point, bounds):
        """
        Projects point within bounds, subroutine for nelder_mead
        """
        sol = [min(max(point[0], bounds[0][0]), bounds[0][1]),
               min(max(point[1], bounds[1][0]), bounds[1][1])]
        
        return sol
    

    #####
    def function_neldermead(self, f_opt, 
                                  bounds=[[-np.inf, np.inf], [-np.inf, np.inf]],
                                  thresh=1e-8, max_iter=500):
        """
        Manually implements nelder mead algorithm with bounds as specified
        """
        # initialize simplex, right angle based
        ws = np.array([[0.01,1], [.01,5], [.21,1]])
        
        # transformation parameters        
        alpha=1
        beta=1/2
        gamma=2
        delta=1/2
        terminate=False
        
        iter_num=0
        while True:
            iter_num += 1
            f_ws = np.array([f_opt(x) for x in ws])
            sorted_args = np.argsort(f_ws)
            ws = ws[sorted_args] # sort working simplex based on f values
            xl,xs,xh = ws
            fl,fs,fh = f_ws[sorted_args]
            
            f_deviation = np.std(f_ws)
            terminate = f_deviation<thresh or iter_num>max_iter
            if terminate:
                break
            
            centroid = (xl+xs)/2
            
            # reflection
            xr = centroid+alpha*(centroid-xh)
            fr = f_opt(xr)
            if fl<=fr<fs:
                ws[2] = self.function_bound_points(xr,bounds)
                continue

            # expansion
            if fr<fl:
                xe = centroid+gamma*(xr-centroid)
                fe = f_opt(xe)
                if fe<fr:
                    ws[2] = self.function_bound_points(xe,bounds)
                    continue
                else:
                    ws[2] = self.function_bound_points(xr,bounds)
                    continue    

            # contraction
            if fr>=fs:
                if fs<=fr<fh:
                    xc = centroid+beta*(xr-centroid)
                    fc = f_opt(xc)
                    if fc<=fr:
                        ws[2] = self.function_bound_points(xc,bounds)
                        continue
                else:
                    xc = centroid+beta*(xh-centroid)
                    fc = f_opt(xc)
                    if fc<fh:
                        ws[2] = self.function_bound_points(xc,bounds)
                        continue
            # shrink
            ws[1] = self.function_bound_points(xl+delta*(ws[1]-xl),bounds)
            ws[2] = self.function_bound_points(xl+delta*(ws[2]-xl),bounds)
            
        return np.mean(ws,axis=0)
    
    
    ##### 
    def function_estimation_error(self, opt_result, i):
        """
        Estimate estimation error of a lineage for optimization
        """ 
        s_opt, tau_opt = opt_result[i,0], opt_result[i,1]
        d_s, d_tau = 1e-8, 1e-5
        
        #-np.exp(-*)
        f_zero = self.function_likelihood_lineage_adaptive([s_opt, tau_opt])
        
        f_plus_s = self.function_likelihood_lineage_adaptive([s_opt + d_s, tau_opt])
        f_minus_s = self.function_likelihood_lineage_adaptive([s_opt - d_s, tau_opt])
    
        f_plus_tau = self.function_likelihood_lineage_adaptive([s_opt, tau_opt + d_tau])
        f_minus_tau = self.function_likelihood_lineage_adaptive([s_opt, tau_opt - d_tau])
    
        f_plus_s_tau = self.function_likelihood_lineage_adaptive([s_opt + d_s, tau_opt + d_tau])
    
        f_ss = (f_plus_s + f_minus_s - 2*f_zero)/d_s**2
        f_tt = (f_plus_tau + f_minus_tau - 2*f_zero)/d_tau**2
        f_st = (f_plus_s_tau - f_plus_s - f_plus_tau + f_zero)/d_s/d_tau
    
        curvature_matrix = np.array([[f_ss,f_st], [f_st,f_tt]]) 
        eigs, eigvecs = np.linalg.eig(curvature_matrix)
        v1, v2 = eigvecs[:,0], eigvecs[:,1]
        lambda1, lambda2 = np.abs(eigs[0]), np.abs(eigs[1])
        
        error_s_lineage =  max(np.abs(v1[0]/np.sqrt(lambda1)), np.abs(v2[0]/np.sqrt(lambda2)))
        error_tau_lineage = max(np.abs(v1[1]/np.sqrt(lambda1)), np.abs(v2[1]/np.sqrt(lambda2)))

        return error_s_lineage, error_tau_lineage


    #####
    def function_fitness_optimization(self, i):
        """
        Estimate fitness and establishment time of adaptive mutation of a lineage using optimization
        """
        self.read_num_seq_lineage = self.read_num_seq[i, :]
        self.cell_num_seq_lineage = self.cell_num_seq[i, :]

        if self.opt_algorithm == 'differential_evolution':
            opt_output = differential_evolution(func = self.function_likelihood_lineage_adaptive,
                                                seed = 1,
                                                bounds = self.bounds_1)
            return opt_output.x
                    
        elif self.opt_algorithm == 'nelder_mead':
            opt_output = self.function_neldermead(self.function_likelihood_lineage_adaptive,
                                                  #x0 = self.optimal_grid[i,:],
                                                  bounds = self.bounds_1,
                                                  thresh = 1e-13,
                                                  max_iter = 500)
            return opt_output


    #####
    def function_run_iteration(self):
        """
        run a single interation
        """
        self.result_s = np.zeros(self.lineages_num)
        self.result_tau = np.zeros(self.lineages_num)
        self.result_log_likelihood_adaptive = np.zeros(self.lineages_num)
        self.result_log_likelihood_neutral = np.zeros(self.lineages_num)

        self.mutant_fraction_numerator = np.zeros(self.seq_num)
        self.cell_depth_seq_theory = np.zeros(self.seq_num)
        self.mutant_cell_num_seq_theory = np.zeros(np.shape(self.read_num_seq))
        self.error_s = np.zeros(self.lineages_num)
        self.error_tau = np.zeros(self.lineages_num)
    
        # Optimization (for all lineages)
        if self.parallelize:
            pool_obj = Pool()            
            opt_result = pool_obj.map(self.function_fitness_optimization, tqdm(range(self.lineages_num)))
            pool_obj.close()
        else:
            opt_result = []
            for i in range(int(self.lineages_num)):
                opt_result.append(self.function_fitness_optimization(i))
        
        opt_result = np.array(opt_result)

        
        for i in range(int(self.lineages_num)):
            self.read_num_seq_lineage = self.read_num_seq[i, :]
            self.cell_num_seq_lineage = self.cell_num_seq[i, :]
            
            #self.result_log_likelihood_adaptive[i] = -self.function_likelihood_lineage_adaptive([np.max([opt_result[i,0], 1e-6]), opt_result[i,1]])
            self.result_log_likelihood_adaptive[i] = -self.function_likelihood_lineage_adaptive([opt_result[i,0], opt_result[i,1]])
            self.result_log_likelihood_neutral[i] = self.function_likelihood_lineage_neutral([0, 0])
            
            #  cell number
            self.cell_depth_seq_theory += self.cell_num_seq_lineage
            # decide whether adaptive or not (needed to calculate mean fitness)
            if self.result_log_likelihood_adaptive[i] - self.result_log_likelihood_neutral[i] > 0:
                _, mutant_cell_num_seq_lineage_theory_adaptive = self.function_cell_num_theory_lineage(opt_result[i,0], opt_result[i,1])
                self.mutant_fraction_numerator += mutant_cell_num_seq_lineage_theory_adaptive
                self.mutant_cell_num_seq_theory[i,:] = mutant_cell_num_seq_lineage_theory_adaptive

                self.error_s[i], self.error_tau[i] = self.function_estimation_error(opt_result, i)
                
        return opt_result
    
    
    #####
    def function_save_data(self):
        """
        Save data according to label: if it's saving a step or the final data
        """
        result_output = {'Fitness': self.result_s,
                         'Establishment_Time': self.result_tau,
                         'Error_Fitness': self.error_s,
                         'Error_Establishment_Time': self.error_tau,
                         'Likelihood_Log': self.result_log_likelihood,
                         'Likelihood_Log_Adaptive': self.result_log_likelihood_adaptive,
                         'Likelihood_Log_Neutral': self.result_log_likelihood_neutral,
                         'Mean_Fitness': self.x_mean_seq[1:],
                         'Kappa_Value': self.kappa_seq, 
                         'Mutant_Cell_Fraction': self.mutant_fraction,
                         'Inference_Time': [self.inference_timing]}
        
        tmp = list(itertools.zip_longest(*list(result_output.values())))
        with open(self.output_filename + '_MutSeq_Result.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(result_output.keys())
            w.writerows(tmp)
    
        tmp = pd.DataFrame(self.mutant_cell_num_seq_theory.astype(int))
        tmp.to_csv(self.output_filename + '_Cell_Number_Mutant_Estimated.csv',
                   index=False, header=False)
    
        tmp = pd.DataFrame(self.cell_num_seq.astype(int))
        tmp.to_csv(self.output_filename + '_Cell_Number.csv', 
                   index=False, header=False)


   #####
    def function_main(self):
        """
        To add
        """
        start = time.time()
        
        # calculate the mean fitness and kappa value
        self.function_mean_fitness_kappa()
        #print(self.x_mean_seq)
        #print(self.kappa_seq)
        
        self.function_sum_term()
        
        opt_result = self.function_run_iteration()
        
        self.mutant_fraction = self.mutant_fraction_numerator/self.cell_depth_seq_theory    
        self.result_log_likelihood = self.result_log_likelihood_adaptive - self.result_log_likelihood_neutral
        self.idx_adaptive_inferred = np.where(self.result_log_likelihood > 0)[0]
        
        #####
        # number of adaptive lineages
        print(len(self.idx_adaptive_inferred))
        #####

        self.result_s[self.idx_adaptive_inferred] = opt_result[self.idx_adaptive_inferred,0]
        self.result_tau[self.idx_adaptive_inferred] = opt_result[self.idx_adaptive_inferred,1]
    
        end = time.time()
        self.inference_timing = np.round(end - start, 5)
        print(f'Total computing time: {self.inference_timing} seconds',flush=True)
        
        self.function_save_data()
