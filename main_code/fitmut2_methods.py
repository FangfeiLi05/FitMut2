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
    def __init__(self, r_seq,
                       t_list,
                       cell_depth_list,
                       Ub,
                       delta_t, 
                       c,
                       opt_algorithm,
                       max_iter_num,
                       parallelize,
                       save_steps,
                       output_filename):
        
        # preparing inputs
        self.r_seq = r_seq
        self.read_depth_seq = np.sum(self.r_seq, axis=0)
        self.lineages_num = np.shape(self.r_seq)[0]
        self.t_list = t_list
        self.seq_num = len(self.t_list)
        self.cell_depth_list = cell_depth_list
        self.ratio = self.read_depth_seq/self.cell_depth_list
        self.n_seq = self.r_seq / self.ratio

        # eliminates zeros from data for later convenience -- also have to modify theoretical model
        # in order to not classify neutrals as adaptive
        # It should be possible to modify this ad hoc choice since we can treat the r_theory=0 case
        # of the bessel function separately when calculating log likelihood of a trajectory.
        self.n_seq[self.n_seq < 1] = 1 
        self.r_seq[self.r_seq < 1] = 1

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
            self.bounds = Bounds([1e-8, -100], [.5, math.floor(self.t_list[-1] - 1)])
        elif self.opt_algorithm == 'nelder_mead':
            self.bounds = [[1e-8, .5], [-100, math.floor(self.t_list[-1] - 1)]]
        
        
        # define other variables
        self.s_mean_seq_dict = dict() # mean fitness at each iteration
        self.mutant_fraction_dict = dict() # fraction of mutatant cells at each iteration
        
        self.iteration_stop_threhold = 5e-7 # threshold for terminating iterations
        self.threshold_adaptive = 0.9 # threshold for determining an adaptive lineage
        self.iter_timing_list = [] # running time for each iteration

        # define some variables for convenient vectorization computing
        self.s_stepsize = 0.02
        self.tau_stepsize = 5
        
        self.s_bin = np.arange(0, 0.4, self.s_stepsize)
        self.s_bin[0] = 1e-8
        if len(self.s_bin)%2 == 0:
            self.s_bin = self.s_bin[:-1]

        self.tau_bin = np.arange(-100, self.t_list[-1]-1, self.tau_stepsize)
        if len(self.tau_bin)%2 == 0:
            self.tau_bin = self.tau_bin[:-1]
        
        self.s_coeff = np.array([1] + [4, 2] * int((len(self.s_bin)-3)/2) + [4, 1])
        self.tau_coeff = np.array([1] + [4, 2] * int((len(self.tau_bin)-3)/2) + [4, 1])
             
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
        Calculate log of the prior (exponential) distribution for mu(s). Returns a 2D array.
        """
        s_len = len(s_array)
        tau_len = len(tau_array)
        mu_s_mean = self.mu_s_mean
        joint_dist1 = np.tile(np.log(self.Ub), (s_len, tau_len))
        joint_dist2 = np.tile(np.log(mu_s_mean), (s_len, tau_len))
        joint_dist3 = np.tile(s_array/mu_s_mean, (tau_len, 1))
        joint_dist4 = np.transpose(joint_dist3, (1,0))
        return joint_dist1 - joint_dist2 - joint_dist4
              
    
    def calculate_kappa(self):
        """
        Calculate kappa value for each timepoint by finding 
        mean and variance of distribution of read number for 
        neutral lineages.
        """
        self.kappa_seq = np.nan * np.zeros(self.seq_num, dtype=float)
        self.kappa_seq[0] = 2.5

        for k in range(self.seq_num-1):
            r_t1_left, r_t1_right = 20, 40 # neutral lineages with read numbers in [20, 40)
            r_t2_left, r_t2_right = 0, 4*r_t1_right
        
            kappa = np.nan * np.zeros(r_t1_right - r_t1_left, dtype=float)
                        
            for r_t1 in range(r_t1_left, r_t1_right):
                pos = self.r_seq[:,k] == r_t1
                
                if np.sum(pos)>100:
                    pdf_conditional_measure = np.histogram(self.r_seq[pos, k+1],
                                                           bins=np.arange(r_t2_left, r_t2_right+0.001),
                                                           density=True)[0]
            
                    dist_x = np.arange(r_t2_left, r_t2_right)
                    param_mean = np.matmul(dist_x, pdf_conditional_measure)
                    param_variance = np.matmul((dist_x - param_mean)**2, pdf_conditional_measure)
                    
                    kappa[r_t1 - r_t1_left] = param_variance/(2*param_mean)
                
                if np.sum(~np.isnan(kappa)): # if not all values of kappa are nan
                    self.kappa_seq[k+1] = np.nanmean(kappa)
                         
        pos_nan = np.isnan(self.kappa_seq)
        if np.sum(pos_nan): # if there is nan type in self.kappa_seq
            self.kappa_seq[pos_nan] = np.nanmean(self.kappa_seq) # define the value as the mean of all non-nan
            
        

    ##########
    def calculate_E(self):
        """
        Pre-calculate a term (capturing the decay in lineage size from the mean fitness)
        to reduce calculations in estimating the number of reads.
        """
        self.t_list_extend = np.concatenate((-np.arange(self.delta_t, 100+self.delta_t, self.delta_t)[::-1], self.t_list))
        seq_num_extend = len(self.t_list_extend)
        #self.s_mean_seq_extend = np.concatenate((1e-8 * np.ones(len(self.t_list_extend) - self.seq_num, dtype=float), self.s_mean_seq))
        self.s_mean_seq_extend = np.concatenate((np.zeros(len(self.t_list_extend) - self.seq_num, dtype=float), self.s_mean_seq))

        self.E_extend = np.ones(seq_num_extend, dtype=float)  
        log_E = 0
        for k in range(1, seq_num_extend):
            #log_E += (self.t_list_extend[k] - self.t_list_extend[k-1]) * self.s_mean_seq_extend[k]
            log_E += (self.t_list_extend[k]-self.t_list_extend[k-1]) * (self.s_mean_seq_extend[k] + self.s_mean_seq_extend[k-1])/2
            self.E_extend[k] = np.exp(-log_E)

        self.E_extend_t_list = np.interp(self.t_list, self.t_list_extend, self.E_extend) # from the very beginning to tk
        
        self.E_t_list = np.zeros(self.seq_num-1, dtype=float) # from tkminus1 to tk
        log_E_t = 0
        for k in range(self.seq_num-1):
            log_E_t = (self.t_list[k+1]-self.t_list[k]) * (self.s_mean_seq[k+1] + self.s_mean_seq[k])/2   
            self.E_t_list[k] =  np.exp(-log_E_t)  

    

    ##########
    def establishment_size_scalar(self, s, tau):
        """
        Calculate establishment size of a mutation with fitness effect s and establishment time tau.
        Inputs: s (scalar)
                tau (scalar)
        Output: established_size (scalar)
        """
        s_mean_tau = np.interp(tau, self.t_list_extend, self.s_mean_seq_extend)
        established_size = self.noise_c / np.maximum(s - s_mean_tau, 0.005)

        return established_size

    

    ##########
    def establishment_size_array(self, s_array, tau_array):
        """
        Calculate establishment size of a mutation with fitness effect s and establishment time tau.
        Inputs: s_array (array, vector)
                tau_array (array, vector) 
        Output: established_size (array, 2D matrix)
        """
        s_len = len(s_array)
        tau_len = len(tau_array)

        s_matrix = np.transpose(np.tile(s_array, (tau_len, 1)), (1,0))

        s_mean_tau = np.tile(np.interp(tau_array, self.t_list_extend, self.s_mean_seq_extend), (s_len, 1)) #(s_len, tau_len)
        established_size = self.noise_c / np.maximum(s_matrix - s_mean_tau, 0.005)

        return established_size


  
    ##########
    def n_theory_scalar(self, s, tau):
        """
        Estimate cell number & mutant cell number all time points for a lineage given s and tau. 
        Inputs: s (scalar)
                tau (scalar)  
        Output: {'cell_number': (array, vector), 
                 'mutant_cell_number': (array, vector)}
        """            
        n_obs = self.n_seq_lineage
        
        n_theory = np.zeros(self.seq_num, dtype=float)
        n_theory[0] = n_obs[0]
        mutant_n_theory = np.zeros(self.seq_num, dtype=float)
        unmutant_n_theory = np.zeros(self.seq_num, dtype=float)
        
        established_size = self.establishment_size_scalar(s, tau)
        E_extend_tau = np.interp(tau, self.t_list_extend, self.E_extend)
        
        for k in range(self.seq_num):
            E_tk_minus_tau = self.E_extend_t_list[k]/E_extend_tau
            mutant1 = np.exp(s * (self.t_list[k] - tau))
            mutant2 = established_size*mutant1
            mutant3 = mutant2*E_tk_minus_tau                      
            mutant_n_theory[k] = np.minimum(mutant3, n_obs[k])
            
            unmutant_n_theory[k] = n_obs[k] - mutant_n_theory[k]

            if k > 0:
                E_tk_minus_tkminus1 = self.E_t_list[k-1]
                #E_tk_minus_tkminus1 = self.E_extend_t_list[k] / self.E_extend_t_list[k-1]
                growth_fac = np.exp(s* (self.t_list[k] - self.t_list[k-1]))
                lineage_size0 = unmutant_n_theory[k-1] + mutant_n_theory[k-1]*growth_fac
                n_theory[k] = lineage_size0 * E_tk_minus_tkminus1
            
        return {'cell_number': n_theory,'mutant_cell_number': mutant_n_theory}
    


    ##########
    def n_theory_array(self, s_array, tau_array):
        """
        Estimate cell number & mutant cell number all time points for a lineage given s and tau. 
        Inputs: s_array (array, vector)
                tau_array (array, vector) 
        Output: {'cell_number': (array, 3D matrix), 
                 'mutant_cell_number': (array, 3D matrix)}
        """
        s_len = len(s_array)
        tau_len = len(tau_array)

        s_matrix = np.transpose(np.tile(s_array, (tau_len, 1)), (1,0))
        tau_matrix  = np.tile(tau_array, (s_len, 1))
            
        n_obs = np.tile(self.n_seq_lineage, (s_len, tau_len, 1))
        
        n_theory = np.zeros((s_len, tau_len, self.seq_num), dtype=float)
        n_theory[:,:,0] = n_obs[:,:,0]
        mutant_n_theory = np.zeros((s_len, tau_len, self.seq_num), dtype=float)
        unmutant_n_theory = np.zeros((s_len, tau_len, self.seq_num), dtype=float)
            
        established_size = self.establishment_size_array(s_array, tau_array) #(s_len, tau_len)
        
        E_extend_tau = np.tile(np.interp(tau_array, self.t_list_extend, self.E_extend), (s_len, 1)) #(s_len, tau_len)
        
        for k in range(self.seq_num):
            E_tk_minus_tau = self.E_extend_t_list[k]/E_extend_tau
            mutant1 = np.exp(s_matrix * (self.t_list[k] - tau_matrix))
            mutant2 = established_size*mutant1
            mutant3 = mutant2*E_tk_minus_tau                     
            mutant_n_theory[:,:,k] = np.minimum(mutant3, n_obs[:,:,k])
            
            unmutant_n_theory[:,:,k] = n_obs[:,:,k] - mutant_n_theory[:,:,k]
                
            if k > 0:
                E_tk_minus_tkminus1 = self.E_t_list[k-1]
                #E_tk_minus_tkminus1 = self.E_extend_t_list[k] / self.E_extend_t_list[k-1]
                growth_fac = np.exp(s_matrix * (self.t_list[k] - self.t_list[k-1]))
                lineage_size0 = unmutant_n_theory[:,:,k-1] + mutant_n_theory[:,:,k-1]*growth_fac
                n_theory[:,:,k] = lineage_size0 * E_tk_minus_tkminus1
    
        return {'cell_number': n_theory,'mutant_cell_number': mutant_n_theory}



    ##########
    def loglikelihood_scalar(self, s, tau):
        """
        Calculate log-likelihood value of a lineage given s and tau.
        Inputs: s(scalar)
                tau (scalar) 
        Output: log-likelihood value of all time points (scalar)
        """        
        n_theory = self.n_theory_scalar(s, tau)['cell_number']
        r_theory = n_theory*self.ratio

        # modifies theoretical read number so that one can compare to modified data without zeros
        r_theory[r_theory < 1] = 1
        
        kappa_inverse = 1/self.kappa_seq

        r_obs = self.r_seq_lineage # observed read count
        ive_arg = 2*np.sqrt(r_theory*r_obs)*kappa_inverse
 
        part2 = 1/2 * np.log(r_theory/r_obs)
        part3 = -(r_theory + r_obs)*kappa_inverse
        part4 = np.log(special.ive(1, ive_arg)) + ive_arg

        log_likelihood_seq_lineage = np.log(kappa_inverse) + part2 + part3 + part4
        log_likelihood_lineage = np.sum(log_likelihood_seq_lineage, axis=0)

        return log_likelihood_lineage
        
    
    ##########
    def loglikelihood_array(self, s_array, tau_array):
        """
        Calculate log-likelihood value of a lineage given s and tau.
        Inputs: s_array (array, vector)
                tau_array (array, vector) 
        Output: log-likelihood value of all time points (array, 2D matrix)
        """
        s_len = len(s_array)
        tau_len = len(tau_array)

        n_theory = self.n_theory_array(s_array, tau_array)['cell_number'] #(s_len, tau_len, seq_num)
        r_theory = n_theory*np.tile(self.ratio, (s_len, tau_len, 1))
        
        # modifies theoretical read number so that one can compare to modified data without zeros
        r_theory[r_theory < 1] = 1
        
        kappa_inverse = np.tile(1/self.kappa_seq, (s_len, tau_len, 1))

        r_obs = np.tile(self.r_seq_lineage, (s_len, tau_len, 1))
        r_obs_inverse = np.tile(1/self.r_seq_lineage, (s_len, tau_len, 1))
        ive_arg = 2*np.sqrt(r_theory*r_obs)*kappa_inverse
 
        part2 = 1/2 * np.log(r_theory*r_obs_inverse)
        part3 = -(r_theory + r_obs)*kappa_inverse
        part4 = np.log(special.ive(1, ive_arg)) + ive_arg

        log_likelihood_seq_lineage = np.log(kappa_inverse) + part2 + part3 + part4
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
        n0 = self.n_seq_lineage[0]
        # weights the prior by lineage size and establishment probability

        # exponential prior
        f_s_tau_joint_log = np.log(self.Ub) - np.log(mu_s_mean) - s/mu_s_mean + np.log(s/self.noise_c * n0)
        # erlang prior
        # f_s_tau_joint_log =  np.log(self.Ub) + np.log(4*s/mu_s_mean**2) - 2*s/mu_s_mean + np.log(s*n0)
        # uniform prior
        # f_s_tau_joint_log =  np.log(self.Ub)  - np.log(2*mu_s_mean) + np.log(s*n0)
        
        return self.loglikelihood_scalar(s, tau) + f_s_tau_joint_log

    
    
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
        n0 = self.n_seq_lineage[0]

        # weights the prior by lineage size and establishment probability
        joint_dist_T = np.tile(np.log(s_array/self.noise_c  * n0), (tau_len, 1))
        joint_dist = np.transpose(joint_dist_T, (1,0))
        if not fine:
            f_s_tau_joint_log = self.f_s_tau_joint_log_part + joint_dist  # exponential prior distribution
        else:
            f_s_tau_joint_log = self.f_s_tau_joint_log_part_fine + joint_dist

        return self.loglikelihood_array(s_array, tau_array) + f_s_tau_joint_log


    ##########
    def log_ratio_adaptive_integral(self, s_array, tau_array):
        """
        probability of a lineage trajectory, given an array of s and tau (using integral method)
        output is scalar, given by the probability integrated over a grid of s and tau
        Also returns the indices of s and tau in the input arrays which gave the highest probability
        """
        integrand_log = self.posterior_loglikelihood_array(s_array, tau_array)
        log_amp_factor = -np.max(integrand_log) + 2
        amp_integrand = np.exp(integrand_log + log_amp_factor)

        s_idx,tau_idx = np.unravel_index(np.argmax(integrand_log),np.shape(integrand_log))
        integral_result = np.dot(np.dot(self.s_coeff, amp_integrand), self.tau_coeff)
        amp_integral = integral_result * self.s_stepsize * self.tau_stepsize / 9
        return np.log(amp_integral) - log_amp_factor,s_idx,tau_idx

    

    ##########
    def posterior_loglikelihood_opt(self, x):
        """
        Calculate posterior log-likelihood value of a lineage given s and tau in optimization
        """
        s, tau = np.maximum(x[0], 1e-8), x[1]
        return -self.posterior_loglikelihood_scalar(s, tau) #minimization only in python

    ##########
    def run_parallel(self, i): 
        """
        i: lineage label
        calculate probability first, then for adaptive lineage output optimized s and tau
        """
        self.r_seq_lineage = self.r_seq[i, :]
        self.n_seq_lineage = self.n_seq[i, :]
        
        p_ratio_log_adaptive,s_idx,tau_idx = self.log_ratio_adaptive_integral(self.s_bin, self.tau_bin)
        p_ratio_log_neutral = self.loglikelihood_scalar(0, 0)
        
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
                opt_output = differential_evolution(func = self.posterior_loglikelihood_opt,
                                                    seed = 1,
                                                    bounds = self.bounds,
                                                    x0 = [self.s_bin[s_idx],self.tau_bin[tau_idx]])
                s_opt, tau_opt = opt_output.x[0], opt_output.x[1]

            elif self.opt_algorithm == 'nelder_mead': 
                opt_output =self.nelder_mead(self.posterior_loglikelihood_opt, 
                                                     bounds = self.bounds,
                                                     thresh = 1e-13,
                                                     max_iter = 500,
                                                     x0 = [self.s_bin[s_idx],self.tau_bin[tau_idx]])
                s_opt, tau_opt = opt_output[0], opt_output[1]
            #elif self.opt_algorithm == 'nelder_mead': 
            #    opt_output = minimize(self.posterior_loglikelihood_opt, 
            #                          x0=[self.s_bin[s_idx],self.tau_bin[tau_idx]],
            #                          method='Nelder-Mead',
            #                          bounds=self.bounds, 
            #                          options={'ftol': 1e-8, 'disp': False, 'maxiter': 500})
            #    s_opt, tau_opt = opt_output.x[0], opt_output.x[1]

        else:
            s_opt, tau_opt = 0, 0
                
        return [p_adaptive, s_opt, tau_opt]

  

                        
    ##########
    def bound_points(self, point, bounds):
        """
        Projects point within bounds, subroutine for nelder_mead
        """
        sol = [min(max(point[0], bounds[0][0]), bounds[0][1]),
               min(max(point[1], bounds[1][0]), bounds[1][1])]
        
        return sol
    

    
    ##########
    def nelder_mead(self, f_opt,bounds=[[-np.inf,np.inf],[-np.inf,np.inf]],thresh=1e-8, max_iter=500,x0=None):
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
                ws[2] = self.bound_points(xr,bounds)
                continue

            # expansion
            if fr<fl:
                xe = centroid+gamma*(xr-centroid)
                fe = f_opt(xe)
                if fe<fr:
                    ws[2] = self.bound_points(xe,bounds)
                    continue
                else:
                    ws[2] = self.bound_points(xr,bounds)
                    continue    

            # contraction
            if fr>=fs:
                if fs<=fr<fh:
                    xc = centroid+beta*(xr-centroid)
                    fc = f_opt(xc)
                    if fc<=fr:
                        ws[2] = self.bound_points(xc,bounds)
                        continue
                else:
                    xc = centroid+beta*(xh-centroid)
                    fc = f_opt(xc)
                    if fc<fh:
                        ws[2] = self.bound_points(xc,bounds)
                        continue
            # shrink
            ws[1] = self.bound_points(xl+delta*(ws[1]-xl),bounds)
            ws[2] = self.bound_points(xl+delta*(ws[2]-xl),bounds)
            
        return np.mean(ws,axis=0)

    
    
    ##########
    def estimation_error_lineage(self, s_opt, tau_opt):
        """
        Estimate estimation error of a lineage for optimization
        """
        d_s, d_tau = 1e-8, 1e-5
    
        f_zero = self.posterior_loglikelihood_opt([s_opt, tau_opt])
        
        f_plus_s = self.posterior_loglikelihood_opt([s_opt + d_s, tau_opt])
        f_minus_s = self.posterior_loglikelihood_opt([s_opt - d_s, tau_opt])
    
        f_plus_tau = self.posterior_loglikelihood_opt([s_opt, tau_opt + d_tau])
        f_minus_tau = self.posterior_loglikelihood_opt([s_opt, tau_opt - d_tau])
    
        f_plus_s_tau = self.posterior_loglikelihood_opt([s_opt + d_s, tau_opt + d_tau])
    
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
    def estimation_error(self):
        for i in self.idx_adaptive_inferred_index:
            self.r_seq_lineage = self.r_seq[i, :]
            self.n_seq_lineage = self.n_seq[i, :]
                
            s_opt = self.result_s[i]
            tau_opt = self.result_tau[i]
            self.error_s[i], self.error_tau[i] = self.estimation_error_lineage(s_opt, tau_opt)
    

    ##########
    def update_mean_fitness(self, k_iter):
        """
        Updated mean fitness & mutant fraction
        """
        self.mutant_fraction_numerator = np.zeros(self.seq_num, dtype=float)
        self.s_mean_numerator = np.zeros(self.seq_num, dtype=float)
        self.mutant_n_seq_theory = np.zeros(np.shape(self.r_seq), dtype=float)
       
        for i in self.idx_adaptive_inferred_index:
            self.r_seq_lineage = self.r_seq[i, :]
            self.n_seq_lineage = self.n_seq[i, :]
            self.mutant_n_seq_theory[i,:] = self.n_theory_scalar(self.result_s[i], self.result_tau[i])['mutant_cell_number']
            self.s_mean_numerator += self.mutant_n_seq_theory[i,:] * self.result_s[i]
            self.mutant_fraction_numerator += self.mutant_n_seq_theory[i,:]
        
        self.s_mean_seq_dict[k_iter] = self.s_mean_numerator/self.cell_depth_list
        self.mutant_fraction_dict[k_iter] = self.mutant_fraction_numerator/self.cell_depth_list

    
    ##########
    def run_iteration(self):
        """
        run a single interation
        """
        # Calculate proability for each lineage to find adaptive lineages, 
        # Then run optimization for adaptive lineages to find their optimized s & tau for adaptive lineages
        if self.parallelize:
            pool_obj = Pool() # might need to change processes=8
            output0 = pool_obj.map(self.run_parallel, tqdm(range(self.lineages_num)))
            pool_obj.close()
            output = np.array(output0)
            self.result_probability_adaptive = np.array(output[:,0])
            self.result_s = np.array(output[:,1])
            self.result_tau = np.array(output[:,2])

        else:
            self.result_probability_adaptive = np.zeros(self.lineages_num, dtype=float)
            self.result_s = np.zeros(self.lineages_num, dtype=float)
            self.result_tau = np.zeros(self.lineages_num, dtype=float)
            for i in range(self.lineages_num):
                output = self.run_parallel(i)
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
    def save_data(self, k_iter, output_label, output_cell_number):
        """
        Save data according to label: if it's saving a step or the final data
        """
        result_output = {'Fitness': self.result_s,
                         'Establishment_Time': self.result_tau,
                         'Error_Fitness': self.error_s,
                         'Error_Establishment_Time': self.error_tau,
                         'Probability_Adaptive': self.result_probability_adaptive,
                         'Mean_Fitness': self.s_mean_seq_dict[k_iter],
                         'Kappa_Value': self.kappa_seq,
                         'Mutant_Cell_Fraction': self.mutant_fraction_dict[k_iter],
                         'Inference_Time': self.iter_timing_list}
        
        to_write = list(itertools.zip_longest(*list(result_output.values())))
        with open(self.output_filename + output_label + '_MutSeq_Result.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(result_output.keys())
            w.writerows(to_write)
        
        to_write = list(itertools.zip_longest(*list(self.s_mean_seq_dict.values())))
        with open(self.output_filename + output_label + '_Mean_fitness_Result.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(self.s_mean_seq_dict.keys())
            w.writerows(to_write)
            
        if output_cell_number == True:
            to_write = pd.DataFrame(self.mutant_n_seq_theory.astype(int))
            to_write.to_csv(self.output_filename + output_label + '_Cell_Number_Mutant_Estimated.csv',
                       index=False, header=False)

            to_write = pd.DataFrame(self.n_seq.astype(int))
            to_write.to_csv(self.output_filename + output_label + '_Cell_Number.csv',
                       index=False, header=False)
   


    #####
    def main(self):
        """
        main function
        """
        start = time.time()
        self.calculate_error = False
        
        self.calculate_kappa()

        for k_iter in range(1, self.max_iter_num+1):
            start_iter = time.time()
            print(f'--- iteration {k_iter} ...')
               
            if k_iter == 1:
                self.s_mean_seq = np.zeros(self.seq_num, dtype=float)
            else:
                self.s_mean_seq = self.s_mean_seq_dict[k_iter-1]
            
        
            self.calculate_E()
            self.run_iteration()
            self.update_mean_fitness(k_iter)

            if self.save_steps == True:
                output_label = '_intermediate_s_' + str(k_iter)
                output_cell_number = False
                self.save_data(k_iter, output_label, output_cell_number)      
                    
            if k_iter > 1:
               stop_check = np.sum((self.s_mean_seq_dict[k_iter] - self.s_mean_seq_dict[k_iter-1])**2)
               print(stop_check)
               if stop_check < self.iteration_stop_threhold:
                   break
                
            end_iter = time.time()
            iter_timing = np.round(end_iter - start_iter, 5)
            self.iter_timing_list.append(iter_timing)
            print(f'    computing time: {iter_timing} seconds', flush=True)
        
        output_label = ''
        output_cell_number = True
        self.estimation_error()
        self.save_data(k_iter, output_label, output_cell_number)
        
        end = time.time()
        inference_timing = np.round(end - start, 5)
        print(f'Total computing time: {inference_timing} seconds',flush=True)
