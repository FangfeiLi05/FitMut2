#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy import special
#from scipy.stats import erlang
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
                       opt_algorithm,
                       max_iter_num,
                       parallelize,
                       save_steps,
                       output_filename):
        
        # preparing inputs
        self.read_num_seq = read_num_seq
        self.read_depth_seq = np.sum(self.read_num_seq, axis=0)
        self.lineages_num = np.shape(self.read_num_seq)[0]
        self.t_seq = t_seq
        self.seq_num = len(self.t_seq)
        self.cell_depth_seq = cell_depth_seq
        self.ratio = np.true_divide(self.read_depth_seq, self.cell_depth_seq)
        self.cell_num_seq = self.read_num_seq / self.ratio
        self.cell_num_seq[self.cell_num_seq < 1] = 1 
        self.read_num_seq[self.read_num_seq < 1] = 1 # approximate distribution is not accurate when r < 1

        self.Ub = Ub
        self.delta_t = delta_t
        self.noise_c = c # noise per generation, effective
        
        self.opt_algorithm = opt_algorithm
        self.max_iter_num = max_iter_num
        self.parallelize = parallelize
        #self.parallelize = False

        self.save_steps = save_steps   
        self.output_filename = output_filename


        # set bounds for the optimization
        if self.opt_algorithm == 'differential_evolution':
            self.bounds = Bounds([1e-8, -100], [.5, math.floor(self.t_seq[-1] - 1)])
        elif self.opt_algorithm == 'nelder_mead':
            self.bounds = [[1e-8, .5], [-100, math.floor(self.t_seq[-1] - 1)]]
        
        
        # define other variables
        self.x_mean_seq_dict = dict() # mean fitness at each iteration
        self.mutant_fraction_dict = dict() # fraction of mutatant cells at each iteration
        
        self.iteration_stop_threhold = 5e-7 # threshold for terminating iterations
        self.threshold_adaptive = 0.5 # threshold for determining an adaptive lineage
        self.iter_timing_list = [] # running time for each iteration

        # define some variables for convinent vectorization computing
        self.s_stepsize = 0.02
        self.tau_stepsize = 5
        
        self.s_bin = np.arange(0, 0.4, self.s_stepsize)
        self.s_bin[0] = 1e-8
        if len(self.s_bin)%2 == 0:
            self.s_bin = self.s_bin[:-1]

        self.tau_bin = np.arange(-100, self.t_seq[-1]-1, self.tau_stepsize)
        if len(self.tau_bin)%2 == 0:
            self.tau_bin = self.tau_bin[:-1]
        
        self.s_coefficient = np.array([1] + [4, 2] * int((len(self.s_bin)-3)/2) + [4, 1])
        self.tau_coefficient = np.array([1] + [4, 2] * int((len(self.tau_bin)-3)/2) + [4, 1])
             
        # part of joint distribution       
        self.mu_s_mean = 0.1
        self.f_s_tau_joint_log_part = self.get_log_prior_mu(self.s_bin,self.tau_bin)

        # finer grids for direct seach of parameters
        self.s_bin_fine = np.arange(0,self.s_bin[-1],.005)
        self.s_bin_fine[0] = 1e-8
        self.tau_bin_fine = np.arange(self.tau_bin[0],self.tau_bin[-1],2)
        self.f_s_tau_joint_log_part_fine = self.get_log_prior_mu(self.s_bin_fine,self.tau_bin_fine)

    ##########
    def get_log_prior_mu(self,s_array,tau_array):
        """
        Calculate log of the prior (exponential) disribution for mu(s). Returns a 2D array.
        """
        s_len = len(s_array)
        tau_len = len(tau_array)
        mu_s_mean = self.mu_s_mean
        joint_tmp1 = np.tile(np.log(self.Ub), (s_len, tau_len))
        joint_tmp2 = np.tile(np.log(mu_s_mean), (s_len, tau_len))
        joint_tmp3 = np.tile(s_array/mu_s_mean, (tau_len, 1))
        joint_tmp4 = np.transpose(joint_tmp3, (1,0))
        return joint_tmp1 - joint_tmp2 - joint_tmp4
              
    
    def function_kappa(self):
        """
        Calculate kappa value for each timepoint by finding 
        mean and variance of distribution of read number for 
        neutral lineages.
        """
        self.kappa_seq = np.nan * np.zeros(self.seq_num, dtype=float)
        self.kappa_seq[0] = 2.5

        for k in range(self.seq_num-1):
            read_num_t1_left, read_num_t1_right = 20, 40 # neutral lineages with read numbers in [20, 40)
            read_num_t2_left, read_num_t2_right = 0, 4*read_num_t1_right
        
            kappa = np.nan * np.zeros(read_num_t1_right - read_num_t1_left, dtype=float)
                        
            for read_num_t1 in range(read_num_t1_left, read_num_t1_right):
                pos = self.read_num_seq[:,k] == read_num_t1
                
                if np.sum(pos)>100:
                    pdf_conditional_measure = np.histogram(self.read_num_seq[pos, k+1],
                                                           bins=np.arange(read_num_t2_left, read_num_t2_right+0.001),
                                                           density=True)[0]
            
                    dist_x = np.arange(read_num_t2_left, read_num_t2_right)
                    param_mean = np.matmul(dist_x, pdf_conditional_measure)
                    param_variance = np.matmul((dist_x - param_mean)**2, pdf_conditional_measure)
                    
                    kappa[read_num_t1 - read_num_t1_left] = param_variance/(2*param_mean)
                
                if np.sum(~np.isnan(kappa)): # if not all values of kappa are nan
                    self.kappa_seq[k+1] = np.nanmean(kappa)
                         
        pos_nan = np.isnan(self.kappa_seq)
        if np.sum(pos_nan): # if there is nan type in self.kappa_seq
            self.kappa_seq[pos_nan] = np.nanmean(self.kappa_seq) # define the value as the mean of all non-nan
            
        

    ##########
    def function_sum_term(self):
        """
        Pre-calculate a term (i.e. sum_term) to reduce calculations in estimating the number of reads.
        """
        self.t_seq_extend = np.concatenate((-np.arange(self.delta_t, 100+self.delta_t, self.delta_t)[::-1], self.t_seq))
        seq_num_extend = len(self.t_seq_extend)
        #self.x_mean_seq_extend = np.concatenate((1e-8 * np.ones(len(self.t_seq_extend) - self.seq_num, dtype=float), self.x_mean_seq))
        self.x_mean_seq_extend = np.concatenate((np.zeros(len(self.t_seq_extend) - self.seq_num, dtype=float), self.x_mean_seq))

        self.sum_term_extend = np.ones(seq_num_extend, dtype=float)  
        sum_term_extend_tmp = 0
        for k in range(1, seq_num_extend):
            #sum_term_extend_tmp += (self.t_seq_extend[k] - self.t_seq_extend[k-1]) * self.x_mean_seq_extend[k]
            sum_term_extend_tmp += (self.t_seq_extend[k]-self.t_seq_extend[k-1]) * (self.x_mean_seq_extend[k] + self.x_mean_seq_extend[k-1])/2
            self.sum_term_extend[k] = np.exp(-sum_term_extend_tmp)

        self.sum_term_extend_t_seq = np.interp(self.t_seq, self.t_seq_extend, self.sum_term_extend) # from the very beginning to tk
        
        self.sum_term_t_seq = np.zeros(self.seq_num-1, dtype=float) # from tkminus1 to tk
        sum_term_extend_t_seq_tmp = 0
        for k in range(self.seq_num-1):
            sum_term_extend_t_seq_tmp = (self.t_seq[k+1]-self.t_seq[k]) * (self.x_mean_seq[k+1] + self.x_mean_seq[k])/2   
            self.sum_term_t_seq[k] =  np.exp(-sum_term_extend_t_seq_tmp)  

    

    ##########
    def function_establishment_size_scalar(self, s, tau):
        """
        Calculate establishment size of a mutation with fitness effect s and establishment time tau.
        Inputs: s (scalar)
                tau (scalar)
        Output: established_size (scalar)
        """
        x_mean_tau = np.interp(tau, self.t_seq_extend, self.x_mean_seq_extend)
        established_size = self.noise_c / np.maximum(s - x_mean_tau, 0.005)

        return established_size

    

    ##########
    def function_establishment_size_array(self, s_array, tau_array):
        """
        Calculate establishment size of a mutation with fitness effect s and establishment time tau.
        Inputs: s_array (array, vector)
                tau_array (array, vector) 
        Output: established_size (array, 2D matrix)
        """
        s_len = len(s_array)
        tau_len = len(tau_array)

        s_matrix_tmp = np.tile(s_array, (tau_len, 1))        
        s_matrix = np.transpose(s_matrix_tmp, (1,0))

        x_mean_tau = np.tile(np.interp(tau_array, self.t_seq_extend, self.x_mean_seq_extend), (s_len, 1)) #(s_len, tau_len)
        established_size = self.noise_c / np.maximum(s_matrix - x_mean_tau, 0.005)

        return established_size


  
    ##########
    def function_cell_num_theory_scalar(self, s, tau):
        """
        Estimate cell number & mutant cell number all time points for a lineage given s and tau. 
        Inputs: s (scalar)
                tau (scalar)  
        Output: {'cell_number': (array, vector), 
                 'mutant_cell_number': (array, vector)}
        """            
        cell_num_seq_lineage_observed = self.cell_num_seq_lineage
        
        cell_num_seq_lineage_theory = np.zeros(self.seq_num, dtype=float)
        cell_num_seq_lineage_theory[0] = cell_num_seq_lineage_observed[0]
        mutant_cell_num_seq_lineage_theory = np.zeros(self.seq_num, dtype=float)
        unmutant_cell_num_seq_lineage_theory = np.zeros(self.seq_num, dtype=float)
        
        established_size = self.function_establishment_size_scalar(s, tau)
        sum_term_extend_tau = np.interp(tau, self.t_seq_extend, self.sum_term_extend)
        
        for k in range(self.seq_num):
            sum_term_tk_minus_tau = np.divide(self.sum_term_extend_t_seq[k], sum_term_extend_tau)
            mutant_tmp1 = np.exp(s * (self.t_seq[k] - tau))
            mutant_tmp2 = np.multiply(established_size, mutant_tmp1)
            mutant_tmp3 = np.multiply(mutant_tmp2, sum_term_tk_minus_tau)                      
            mutant_cell_num_seq_lineage_theory[k] = np.minimum(mutant_tmp3, cell_num_seq_lineage_observed[k])
            
            unmutant_cell_num_seq_lineage_theory[k] = cell_num_seq_lineage_observed[k] - mutant_cell_num_seq_lineage_theory[k]

            if k > 0:
                sum_term_tk_minus_tkminus1 = self.sum_term_t_seq[k-1]
                #sum_term_tk_minus_tkminus1 = self.sum_term_extend_t_seq[k] / self.sum_term_extend_t_seq[k-1]
                both_tmp1 = np.exp(s* (self.t_seq[k] - self.t_seq[k-1]))
                both_tmp2 = unmutant_cell_num_seq_lineage_theory[k-1] + np.multiply(mutant_cell_num_seq_lineage_theory[k-1], both_tmp1)
                cell_num_seq_lineage_theory[k] = both_tmp2 * sum_term_tk_minus_tkminus1
            
        output = {'cell_number': cell_num_seq_lineage_theory, 
                  'mutant_cell_number': mutant_cell_num_seq_lineage_theory}

        return output
    


    ##########
    def function_cell_num_theory_array(self, s_array, tau_array):
        """
        Estimate cell number & mutant cell number all time points for a lineage given s and tau. 
        Inputs: s_array (array, vector)
                tau_array (array, vector) 
        Output: {'cell_number': (array, 3D matrix), 
                 'mutant_cell_number': (array, 3D matrix)}
        """
        s_len = len(s_array)
        tau_len = len(tau_array)

        s_matrix_tmp = np.tile(s_array, (tau_len, 1))        
        s_matrix = np.transpose(s_matrix_tmp, (1,0))
        tau_matrix  = np.tile(tau_array, (s_len, 1))
            
        cell_num_seq_lineage_observed = np.tile(self.cell_num_seq_lineage, (s_len, tau_len, 1))
        
        cell_num_seq_lineage_theory = np.zeros((s_len, tau_len, self.seq_num), dtype=float)
        cell_num_seq_lineage_theory[:,:,0] = cell_num_seq_lineage_observed[:,:,0]
        mutant_cell_num_seq_lineage_theory = np.zeros((s_len, tau_len, self.seq_num), dtype=float)
        unmutant_cell_num_seq_lineage_theory = np.zeros((s_len, tau_len, self.seq_num), dtype=float)
            
        established_size = self.function_establishment_size_array(s_array, tau_array) #(s_len, tau_len)
        
        sum_term_extend_tau_tmp = np.interp(tau_array, self.t_seq_extend, self.sum_term_extend)
        sum_term_extend_tau = np.tile(sum_term_extend_tau_tmp, (s_len, 1)) #(s_len, tau_len)
        
        for k in range(self.seq_num):
            sum_term_tk_minus_tau = np.divide(self.sum_term_extend_t_seq[k], sum_term_extend_tau)
            mutant_tmp1 = np.exp(s_matrix * (self.t_seq[k] - tau_matrix))
            mutant_tmp2 = np.multiply(established_size, mutant_tmp1)
            mutant_tmp3 = np.multiply(mutant_tmp2, sum_term_tk_minus_tau)                      
            mutant_cell_num_seq_lineage_theory[:,:,k] = np.minimum(mutant_tmp3, cell_num_seq_lineage_observed[:,:,k])
            
            unmutant_cell_num_seq_lineage_theory[:,:,k] = cell_num_seq_lineage_observed[:,:,k] - mutant_cell_num_seq_lineage_theory[:,:,k]
                
            if k > 0:
                sum_term_tk_minus_tkminus1 = self.sum_term_t_seq[k-1]
                #sum_term_tk_minus_tkminus1 = self.sum_term_extend_t_seq[k] / self.sum_term_extend_t_seq[k-1]
                both_tmp1 = np.exp(s_matrix * (self.t_seq[k] - self.t_seq[k-1]))
                both_tmp2 = unmutant_cell_num_seq_lineage_theory[:,:,k-1] + np.multiply(mutant_cell_num_seq_lineage_theory[:,:,k-1], both_tmp1)
                cell_num_seq_lineage_theory[:,:,k] = both_tmp2 * sum_term_tk_minus_tkminus1
    
        output = {'cell_number': cell_num_seq_lineage_theory,
                  'mutant_cell_number': mutant_cell_num_seq_lineage_theory}

        return output



    ##########
    def prior_loglikelihood_scalar(self, s, tau):
        """
        Calculate log-likelihood value of a lineage given s and tau.
        Inputs: s(scalar)
                tau (scalar) 
        Output: log-likelihood value of all time points (scalar)
        """        
        output = self.function_cell_num_theory_scalar(s, tau)
        cell_num_seq_lineage_theory = output['cell_number']
        read_num_seq_lineage_theory = np.multiply(cell_num_seq_lineage_theory, self.ratio)
        read_num_seq_lineage_theory[read_num_seq_lineage_theory < 1] = 1
        
        tmp_kappa_reverse = 1/self.kappa_seq

        tmp_theory = read_num_seq_lineage_theory
        tmp_observed = self.read_num_seq_lineage
        tmp_observed_reverse = 1/tmp_observed
        ive_ele = 2* np.multiply(np.sqrt(np.multiply(tmp_theory, tmp_observed)), tmp_kappa_reverse)
 
        tmp_part1 = np.log(tmp_kappa_reverse)
        tmp_part2 = 1/2 * np.log(np.multiply(tmp_theory, tmp_observed_reverse))
        tmp_part3 = -np.multiply(tmp_theory + tmp_observed, tmp_kappa_reverse)
        tmp_part4 = np.log(special.ive(1, ive_ele)) + ive_ele

        log_likelihood_seq_lineage = tmp_part1 + tmp_part2 + tmp_part3 + tmp_part4
        log_likelihood_lineage = np.sum(log_likelihood_seq_lineage, axis=0)

        return log_likelihood_lineage
        
    
    ##########
    def prior_loglikelihood_array(self, s_array, tau_array):
        """
        Calculate log-likelihood value of a lineage given s and tau.
        Inputs: s_array (array, vector)
                tau_array (array, vector) 
        Output: log-likelihood value of all time poins (array, 2D matrix)
        """
        s_len = len(s_array)
        tau_len = len(tau_array)

        output = self.function_cell_num_theory_array(s_array, tau_array)
        cell_num_seq_lineage_theory = output['cell_number'] #(s_len, tau_len, seq_num)
        read_num_seq_lineage_theory = np.multiply(cell_num_seq_lineage_theory, np.tile(self.ratio, (s_len, tau_len, 1)))
        read_num_seq_lineage_theory[read_num_seq_lineage_theory < 1] = 1
        
        tmp_kappa_reverse = np.tile(1/self.kappa_seq, (s_len, tau_len, 1))

        tmp_theory = read_num_seq_lineage_theory
        tmp_observed = np.tile(self.read_num_seq_lineage, (s_len, tau_len, 1))
        tmp_observed_reverse = np.tile(1/self.read_num_seq_lineage, (s_len, tau_len, 1))
        ive_ele = 2* np.multiply(np.sqrt(np.multiply(tmp_theory, tmp_observed)), tmp_kappa_reverse)
 
        tmp_part1 = np.log(tmp_kappa_reverse)
        tmp_part2 = 1/2 * np.log(np.multiply(tmp_theory, tmp_observed_reverse))
        tmp_part3 = -np.multiply(tmp_theory + tmp_observed, tmp_kappa_reverse)
        tmp_part4 = np.log(special.ive(1, ive_ele)) + ive_ele

        log_likelihood_seq_lineage = tmp_part1 + tmp_part2 + tmp_part3 + tmp_part4
        log_likelihood_lineage = np.sum(log_likelihood_seq_lineage, axis=2)

        return log_likelihood_lineage
    


    ##########
    def posterior_loglikelihood_scalar(self, s, tau):
        """
        Calculate posterior log-likelihood value of a lineage given s and tau.
        Inputs: s (scalar)
                tau (scalar) 
        Output: log-likelihood value of all time poins (scalar)
        """
        mu_s_mean = self.mu_s_mean
        f_s_tau_joint_log = np.log(self.Ub) - np.log(mu_s_mean) - s/mu_s_mean + np.log(s/self.noise_c * self.cell_num_seq_lineage[0])  #exponential (Mathematica)
        #f_s_tau_joint_log =  np.log(self.Ub) + np.log(4*s/mu_s_mean**2) - 2*s/mu_s_mean + np.log(s * self.cell_num_seq_lineage[0])  #erlang prior:
        #f_s_tau_joint_log =  np.log(self.Ub)  - np.log(2*mu_s_mean) + np.log(s * self.cell_num_seq_lineage[0])  #uniform prior
        
        output = self.prior_loglikelihood_scalar(s, tau) + f_s_tau_joint_log
        
        return output

    
    
    ##########
    def posterior_loglikelihood_array(self, s_array, tau_array,fine=False):
        """
        Calculate posterior log-likelihood value of a lineage given s and tau.
        Calculates the log likelihood on a finer grid if specified
        Inputs: s_array (array, vector)
                tau_array (array, vector)
                fine (boolean)
        Output: log-likelihood value of all time points (array, 2D matrix)
        """
        tau_len = len(tau_array)

        joint_tmp5 = np.tile(np.log(s_array/self.noise_c  * self.cell_num_seq_lineage[0]), (tau_len, 1))
        joint_tmp6 = np.transpose(joint_tmp5, (1,0))
        if not fine:
            f_s_tau_joint_log = self.f_s_tau_joint_log_part + joint_tmp6  #exponential prior distribution
        else:
            f_s_tau_joint_log = self.f_s_tau_joint_log_part_fine + joint_tmp6
        output = self.prior_loglikelihood_array(s_array, tau_array) + f_s_tau_joint_log

        return output


    ##########
    def log_ratio_adaptive_integral(self, s_array, tau_array):
        """
        probability of a lineage trajectory, given an array of s and tau (using integral method)
        output is scalar, given by the probability integrated over a grid of s and tau
        Also returns the indices of s and tau in the input arrays which gave the highest probability
        """
        integrand_log = self.posterior_loglikelihood_array(s_array, tau_array)
        amplify_factor_log = -np.max(integrand_log) + 2
        amplify_integrand = np.exp(integrand_log + amplify_factor_log)

        s_idx,tau_idx = np.unravel_index(np.argmax(integrand_log),np.shape(integrand_log))
        tmp2 = np.dot(np.dot(self.s_coefficient, amplify_integrand), self.tau_coefficient)
        amplify_integral = tmp2 * self.s_stepsize * self.tau_stepsize / 9
        output  = np.log(amplify_integral) - amplify_factor_log
        return output,s_idx,tau_idx

    

    ##########
    def function_posterior_loglikelihood_opt(self, x):
        """
        Calculate posterior log-likelihood value of a lineage given s and tau in optimization
        """
        s, tau = np.maximum(x[0], 1e-8), x[1]
        output = self.posterior_loglikelihood_scalar(s, tau)
        return -output #minimization only in python

    ##########
    def function_parallel(self, i): 
        """
        i: lineage label
        calculate probability first, then for adaptive lineage output optimized s and tau
        """
        self.read_num_seq_lineage = self.read_num_seq[i, :]
        self.cell_num_seq_lineage = self.cell_num_seq[i, :]
        
        p_ratio_log_adaptive,s_idx,tau_idx = self.log_ratio_adaptive_integral(self.s_bin, self.tau_bin)
        p_ratio_log_neutral = self.prior_loglikelihood_scalar(0, 0)
        
        p_ratio_log = p_ratio_log_adaptive - p_ratio_log_neutral
        if p_ratio_log <= 40:
            p_ratio = np.exp(p_ratio_log)
            p_adaptive = p_ratio /(1 + p_ratio)
        else:
            p_adaptive = 1

        if p_adaptive > self.threshold_adaptive:
            if self.opt_algorithm == 'direct_search':
                # calculate on a finer grid
                log_likelihood_fine = self.posterior_loglikelihood_array(self.s_bin_fine,self.tau_bin_fine,fine=True) 
                s_idx1,tau_idx1 = np.unravel_index(np.argmax(log_likelihood_fine),np.shape(log_likelihood_fine))
                s_opt, tau_opt = self.s_bin_fine[s_idx1], self.tau_bin_fine[tau_idx1]

            elif self.opt_algorithm == 'differential_evolution':
                opt_output = differential_evolution(func = self.function_posterior_loglikelihood_opt,
                                                    seed = 1,
                                                    bounds = self.bounds,
                                                    x0 = [self.s_bin[s_idx],self.tau_bin[tau_idx]])
                s_opt, tau_opt = opt_output.x[0], opt_output.x[1]

            elif self.opt_algorithm == 'nelder_mead': 
                opt_output =self.function_neldermead(self.function_posterior_loglikelihood_opt, 
                                                     bounds = self.bounds,
                                                     thresh = 1e-13,
                                                     max_iter = 500,
                                                     x0 = [self.s_bin[s_idx],self.tau_bin[tau_idx]])
                s_opt, tau_opt = opt_output[0], opt_output[1]
            #elif self.opt_algorithm == 'nelder_mead': 
            #    opt_output = minimize(self.function_posterior_loglikelihood_opt, 
            #                          x0=[self.s_bin[s_idx],self.tau_bin[tau_idx]],
            #                          method='Nelder-Mead',
            #                          bounds=self.bounds, 
            #                          options={'ftol': 1e-8, 'disp': False, 'maxiter': 500})
            #    s_opt, tau_opt = opt_output.x[0], opt_output.x[1]

        else:
            s_opt, tau_opt = 0, 0
                
        return [p_adaptive, s_opt, tau_opt]

  

                        
    ##########
    def function_bound_points(self, point, bounds):
        """
        Projects point within bounds, subroutine for nelder_mead
        """
        sol = [min(max(point[0], bounds[0][0]), bounds[0][1]),
               min(max(point[1], bounds[1][0]), bounds[1][1])]
        
        return sol
    

    
    ##########
    def function_neldermead(self, f_opt, 
                                  bounds=[[-np.inf,np.inf],[-np.inf,np.inf]],
                                  thresh=1e-8, max_iter=500,x0=None):
        """
        Manually implements nelder mead algorithm with bounds as specified
        """
        if x0 is None:
            ws = np.array([[0.01,1], [.01,5], [.21,1]])
        else:
            xi,yi = x0
            ws = np.array([[xi,yi],[xi,yi+5],[xi+.05,yi]])
        
        # transformation parameters
        alpha = 1
        beta = 1/2
        gamma = 2
        delta = 1/2
        terminate = False
        
        iter_num=0
        while True:
            iter_num+=1
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

    
    
    ##########
    def function_estimation_error_lineage(self, s_opt, tau_opt):
        """
        Estimate estimation error of a lineage for optimization
        """
        d_s, d_tau = 1e-8, 1e-5
    
        f_zero = self.function_posterior_loglikelihood_opt([s_opt, tau_opt])
        
        f_plus_s = self.function_posterior_loglikelihood_opt([s_opt + d_s, tau_opt])
        f_minus_s = self.function_posterior_loglikelihood_opt([s_opt - d_s, tau_opt])
    
        f_plus_tau = self.function_posterior_loglikelihood_opt([s_opt, tau_opt + d_tau])
        f_minus_tau = self.function_posterior_loglikelihood_opt([s_opt, tau_opt - d_tau])
    
        f_plus_s_tau = self.function_posterior_loglikelihood_opt([s_opt + d_s, tau_opt + d_tau])
    
        f_ss = (f_plus_s + f_minus_s - 2*f_zero)/d_s**2
        f_tt = (f_plus_tau + f_minus_tau - 2*f_zero)/d_tau**2
        f_st = (f_plus_s_tau - f_plus_s - f_plus_tau + f_zero)/d_s/d_tau
    
        curvature_matrix = np.array([[f_ss,f_st], [f_st,f_tt]])
        eigs, eigvecs = np.linalg.eig(curvature_matrix)
        v1, v2 = eigvecs[:,0], eigvecs[:,1]
        lambda1, lambda2 = np.abs(eigs[0]), np.abs(eigs[1])
        
        if lambda1==0 or lambda2==0:
            error_s_lineage = np.nan
            error_tau_lineage = np.nan
        else:
            error_s_lineage =  max(np.abs(v1[0]/np.sqrt(lambda1)), np.abs(v2[0]/np.sqrt(lambda2)))
            error_tau_lineage = max(np.abs(v1[1]/np.sqrt(lambda1)), np.abs(v2[1]/np.sqrt(lambda2)))

        return error_s_lineage, error_tau_lineage

    
    ##########
    def function_estimation_error(self):
        for i in self.idx_adaptive_inferred_index:
            self.read_num_seq_lineage = self.read_num_seq[i, :]
            self.cell_num_seq_lineage = self.cell_num_seq[i, :]
                
            s_opt = self.result_s[i]
            tau_opt = self.result_tau[i]
            self.error_s[i], self.error_tau[i] = self.function_estimation_error_lineage(s_opt, tau_opt)
    

    ##########
    def function_update_mean_fitness(self, k_iter):
        """
        Updated mean fitness & mutant fraction
        """
        self.mutant_fraction_numerator = np.zeros(self.seq_num, dtype=float)
        self.x_mean_numerator = np.zeros(self.seq_num, dtype=float)
        self.mutant_cell_num_seq_theory = np.zeros(np.shape(self.read_num_seq), dtype=float)
       
        for i in self.idx_adaptive_inferred_index:
            self.read_num_seq_lineage = self.read_num_seq[i, :]
            self.cell_num_seq_lineage = self.cell_num_seq[i, :]
            output = self.function_cell_num_theory_scalar(self.result_s[i], self.result_tau[i])
            self.mutant_cell_num_seq_theory[i,:] = output['mutant_cell_number']
            self.x_mean_numerator += self.mutant_cell_num_seq_theory[i,:] * self.result_s[i]
            self.mutant_fraction_numerator += self.mutant_cell_num_seq_theory[i,:]
        
        self.x_mean_seq_dict[k_iter] = self.x_mean_numerator/self.cell_depth_seq
        self.mutant_fraction_dict[k_iter] = self.mutant_fraction_numerator/self.cell_depth_seq

    
    ##########
    def function_run_iteration(self):
        """
        run a single interation
        """
        # Calculate proability for each lineage to find adaptive lineages, 
        # Then run optimization for adaptive lineages to find their optimized s & tau for adaptive lineages
        if self.parallelize:
            pool_obj = Pool() # might need to change processes=8
            output_tmp = pool_obj.map(self.function_parallel, tqdm(range(self.lineages_num)))
            pool_obj.close()
            output = np.array(output_tmp)
            self.result_probability_adaptive = np.array(output[:,0])
            self.result_s = np.array(output[:,1])
            self.result_tau = np.array(output[:,2])

        else:
            self.result_probability_adaptive = np.zeros(self.lineages_num, dtype=float)
            self.result_s = np.zeros(self.lineages_num, dtype=float)
            self.result_tau = np.zeros(self.lineages_num, dtype=float)
            for i in range(self.lineages_num):
                output = self.function_parallel(i)
                self.result_probability_adaptive[i] = output[0]
                self.result_s[i] = output[1]
                self.result_tau[i] = output[2]
        
        self.idx_adaptive_inferred_logical = self.result_probability_adaptive > self.threshold_adaptive
        self.idx_adaptive_inferred_index = np.where(self.idx_adaptive_inferred_logical)[0]
        

        #####
        # number of adaptive lineages
        print(len(self.idx_adaptive_inferred_index))
        #####

        self.error_s = np.zeros(self.lineages_num, dtype=float)
        self.error_tau = np.zeros(self.lineages_num, dtype=float)

    
    #####
    def function_save_data(self, k_iter, output_label, output_cell_number):
        """
        Save data according to label: if it's saving a step or the final data
        """
        result_output = {'Fitness': self.result_s,
                         'Establishment_Time': self.result_tau,
                         'Error_Fitness': self.error_s,
                         'Error_Establishment_Time': self.error_tau,
                         'Probability_Adaptive': self.result_probability_adaptive,
                         'Mean_Fitness': self.x_mean_seq_dict[k_iter],
                         'Kappa_Value': self.kappa_seq,
                         'Mutant_Cell_Fraction': self.mutant_fraction_dict[k_iter],
                         'Inference_Time': self.iter_timing_list}
        
        tmp = list(itertools.zip_longest(*list(result_output.values())))
        with open(self.output_filename + output_label + '_MutSeq_Result.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(result_output.keys())
            w.writerows(tmp)
        
        tmp = list(itertools.zip_longest(*list(self.x_mean_seq_dict.values())))
        with open(self.output_filename + output_label + '_Mean_fitness_Result.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(self.x_mean_seq_dict.keys())
            w.writerows(tmp)
            
        if output_cell_number == True:
            tmp = pd.DataFrame(self.mutant_cell_num_seq_theory.astype(int))
            tmp.to_csv(self.output_filename + output_label + '_Cell_Number_Mutant_Estimated.csv',
                       index=False, header=False)
   


    #####
    def function_main(self):
        """
        main function
        """
        start = time.time()
        self.calculate_error = False
        
        self.function_kappa()

        for k_iter in range(1, self.max_iter_num+1):
            start_iter = time.time()
            print(f'--- iteration {k_iter} ...')
               
            if k_iter == 1:
                self.x_mean_seq = np.zeros(self.seq_num, dtype=float)
            else:
                self.x_mean_seq = self.x_mean_seq_dict[k_iter-1]
            
        
            self.function_sum_term()
            self.function_run_iteration()
            self.function_update_mean_fitness(k_iter)

            if self.save_steps == True:
                output_label = '_intermediate_s_' + str(k_iter)
                output_cell_number = False
                self.function_save_data(k_iter, output_label, output_cell_number)      
                    
            if k_iter > 1:
               stop_check = np.sum((self.x_mean_seq_dict[k_iter] - self.x_mean_seq_dict[k_iter-1])**2)
               print(stop_check)
               if stop_check < self.iteration_stop_threhold:
                   break
                
            end_iter = time.time()
            iter_timing = np.round(end_iter - start_iter, 5)
            self.iter_timing_list.append(iter_timing)
            print(f'    computing time: {iter_timing} seconds', flush=True)
        
        output_label = ''
        output_cell_number = True
        self.function_estimation_error()
        self.function_save_data(k_iter, output_label, output_cell_number)
        
        end = time.time()
        inference_timing = np.round(end - start, 5)
        print(f'Total computing time: {inference_timing} seconds',flush=True)
