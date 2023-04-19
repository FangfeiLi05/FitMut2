[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Contact Info](https://img.shields.io/badge/Contact-fangfeili0525@gmail.com-blue.svg)]()


## FitMut2

### 1. What is FitMut2?

FitMut2 is an algorithm developed for identifying adaptive mutations that established in barcoded evolution experiments, and inferring their mutational parameters (fitness effect and establishment time). It is preceded by FitMut1, which was developed in [S. F. Levy, et al. Quantitative evolutionary dynamics using high-resolution lineage tracking. Nature, 519(7542): 181-186 (2015)](https://www.nature.com/articles/nature14279) and originally implemented in Mathematica. In this repository we have reimplemented FitMut1 in Python and additionally adapted it for higher accuracy in situations with lower sequencing coverage. If you use this software, please reference our [preprint](https://www.biorxiv.org/content/10.1101/2022.09.25.509409v1). (Codes and results for this paper are store in a shared folder in Google Drive [here]
(https://drive.google.com/drive/folders/1WNmyXov6y5G-mgrvxVLhTLjZGEI-mgnC?usp=share_link))

FitMut2 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

This repository has two main scripts (aside from the implementation of FitMut1):
* `fitmutsimu_run.py` simulates the experimental process of a barcode-sequencing (bar-seq) evolution experiment. This can be used to test the inference algorithm on simulated data where the ground truth is known.
* `fitmut2_run.py` identifies adaptive mutations (as well as inferring their fitness effects and establishment times) that established in bar-seq evolution experiments from read count time series data.

A walk-through is included as the jupyter notebook [here](https://github.com/FangfeiLi05/FitMut2/blob/master/walk_through/walk_through.ipynb).


### 2. Installing FitMut2

* Python 3 is required. This version has been tested on a MacBook Pro (Apple M1 Chip, 8 GB Memory), with Python 3.8.5.
* Clone this repository by running `git clone https://github.com/FangfeiLi05/FitMut2.git` in terminal.
* `cd` to the root directory of the project (the folder containing `README.md`).
* Install dependencies by running `pip install -r requirements.txt` in terminal.


### 3. How to use FitMut2?

#### 3.1. Evolution Simulation
`fitmutsimu_run.py` simulates the entire experimental process of barcode-sequencing (bar-seq) evolution experiment with serial dilution of a barcoded cell population. This simulation models all sources of noise, including growth noise, noise from cell transfers, DNA extraction, PCR, and sequencing, as Poisson randomness with the appropriate multiplicative factor.

##### Options
* `--lineage_number` or `-l`: number of lineages to simulate. Each lineage begins the evolution experiment with an average size of 100 cells, where the spread is determined by variability in the pregrowth phase.
* `--t_seq` or `-t`: a .csv file, with
  + 1st column: sequenced time points measured in number of generations
  + 2nd+ columns: average number of reads per barcode for each sequenced time point (accepts multiple columns for multiple sequencing replicates with e.g. variable coverage)
* `--mutation_fitness` or `-s`: a .csv file, with
  + 1st column: total beneficial mutation rate, Ub
  + 2nd column: bin edges of the arbitrary DFE
  + 3rd column: normalized counts in each bin of the 2nd column
* `--maximum_mutation_number` or `-max_mut_num`: maximum number of mutations allowed in each single cell (`default: 1`)
* `--t_pregrowth` or `-t_pre`: number of generations in pre-growth (`default: 16`)
* `--cell_num_average_bottleneck` or `-n_b`: average number of cells per barcode transferred at each bottleneck (`default: 100`)
* `--c` or `-c`: half of variance introduced by cell growth and cell transfer (`default: 1`)
* `--dna_copies` or `-d`: average genome template copies per barcode in PCR (`default: 500`)
* `--pcr_cycles` or `-p`: number of cycles in PCR (`default: 25`)
* `--output_filename` or `-o`: prefix of output files (`default: output`)

##### Outputs
* `simu_output_EvoSimulation_Read_Number.csv`: read number per barcode for each time point
* `simu_output_EvoSimulation_Mutation_Info.csv`: information of adaptive mutations that established
* `simu_output_EvoSimulation_Other_Info.csv`: a record of some inputs (also fraction of mutant cells of the population)
* `simu_output_EvoSimulation_Bottleneck_Cell_Number.csv`: bottleneck cell number per barcode for each time point
* `simu_output_EvoSimulation_Bottleneck_Cell_Number_Neutral.csv`: bottleneck neutral cell number per barcode for each time point
* `simu_output_EvoSimulation_Saturated_Cell_Number.csv`: saturated cell number per barcode for each time point
* `simu_output_EvoSimulation_Saturated_Cell_Number_Neutral.csv`: saturated neutral cell number per barcode for each time point


##### For Help
```
python fitmutsimu_run.py --help
```

##### Example Use Case
```
python fitmutsimu_run.py -l 10000 -t simu_input_time_points.csv -s simu_input_mutation_fitness.csv -o test
```    


#### 3.2. Mutation Identification
`fitmut2_run.py` identifies adaptive mutations in barcoded evolution experiments from read-count time series data, and estimates their fitness effects and establishment times. 

##### Options
* `--input` or `-i`: a .csv file, with each column being the read number per barcode at each sequenced time point
* `--t_seq` or `-t`: a .csv file, with
  + 1st column: sequenced time points evaluated in number of generations
  + 2nd column: number of cells transferred at each sequenced time point, multiplied by the time (in generations) between time points. This is what we call _effective cell number_.
* `--mutation_rate` or `-u`: total beneficial mutation rate per generation per cell (default chosen from expectation in _S. cerevisiae_). This choice affects the prior distribution, and using the default value in most cases should be fine. (`default: 1e-5`)
* `--delta_t` or `-dt`: number of generations between bottlenecks. This is approximately given by the logarithm (base 2) of the dilution factor between transfers.  (`default: 8`)
* `--c` or `-c`: half of variance introduced by cell growth and cell transfer. In most cases the default value should suffice, unless the experimental value is measureable. (`default: 1`)
* `--maximum_iteration_number` or `-n`: maximum number of iterations in the self consistent estimation of mean fitness and lineage fitnesses (`default: 50`)
* `--opt_algorithm` or `-a`: optimization algorithm (Nelder Mead or Differential Evolution) (`default: nelder_mead`)
* `--parallelize` or `-p`: whether to use Python multiprocess module to parallelize inference across lineages (`default: True`)
* `--save_steps` or `-s`: whether to save the data files after each iteration of inference (`default: False`)
* `--output_filename` or `-o`: prefix of output files (`default: output`)

##### Outputs
* `output_MutSeq_Result.csv`: a .csv file, with
  + 1st column of .csv: estimated fitness effect of each lineage
  + 2nd column of .csv: estimated establishment time of each lineage
  + 3rd column of .csv: uncertainty in fitness effect
  + 4th column of .csv: uncertainty in establishment time
  + 5th column of .csv: probability of each lineage containing an adaptive mutation
  + 6th column of .csv: estimated mean fitness per sequenced time point
  + 7th column of .csv: estimated kappa (noise parameter, see preprint for definition) per sequenced time point
  + 8th column of .csv: estimated fraction of mutant cells of the population per sequenced time point
* `output_Mean_fitness_Result.csv`: estimated mean fitness at each iteration
* `output_Cell_Number_Mutant_Estimated.csv`: estimated effective number of mutant cells per barcode for each time point
* `output_Cell_Number.csv`: effective cell number per barcode for each time point 

##### For Help
```
python fitmut2_run.py --help
```  

##### Example Use Case
```
python fitmut2_run.py -i simu_test_EvoSimulation_Read_Number.csv -t fitmut_input_time_points.csv -o test
```



