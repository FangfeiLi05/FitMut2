[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Contact Info](https://img.shields.io/badge/Contact-fangfeili0525@gmail.com-blue.svg)]()


## FitMut2

### 1. What is FitMut2?

FitMut2 is an improved version of FitMut1. FitMut1 is a Python reimplementation version of the [Mathematica](https://www.wolfram.com/mathematica/) tool that developed for identifying adaptive mutations that established in barcoded evolution experiments, and inferring their mutational parameters (fitness effect and establishment time) (see more details of FitMut1 in reference: [S. F. Levy, et al. Quantitative evolutionary dynamics using high-resolution lineage tracking. Nature, 519(7542): 181-186 (2015)](https://www.nature.com/articles/nature14279). If you use this software, please reference: [bioRxiv](). FitMut2 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

It currently has two main functions:
* `fitmutsimu_run_v1.py` simulates the entire experimental process of barcode-sequencing (bar-seq) evolution experiment. 
* `fitmut2_run_v1.py` identifies adaptive mutations (as well as inferring their fitness effects and establishment times) that established in bar-seq evolution experiments from read-count time-series data.

A walk-through is included as the jupyter notebook [here](https://github.com/FangfeiLi05/FitMut2/blob/master/walk_through/walk_through.ipynb).


### 2. How to install FitMut2?

* Python 3 is required. This version has been tested on a MacBook Pro (Apple M1 Chip, 8 GB Memeory), with Python 3.8.5.
* Clone this repository by running `git clone https://github.com/FangfeiLi05/FitMut2.git` in terminal.
* `cd` to the root directory of the project (the folder containing `README.md`).
* Install dependencies by running `pip install -r requirements.txt` in terminal.


### 3. How to use FitMut2?

#### 3.1. Evolution Simulation
`fitmutsimu_run_v1.py` simulates the entire experimental process of barcode-sequencing (bar-seq) evolution experiment with serial dilution of a barcoded cell population. This simulation includes all sources of noise, including growth noise, noise from cell transfers, DNA extraction, PCR, and sequencing.

##### Options
* `--linegaes_number` or `-l`: number of lineages
* `--t_seq` or `-t`: a .csv file, with
  + 1st column: sequenced time-points evaluated in number of generations
  + 2nd+ columns: average number of reads per barcode for each sequenced time-point (accept multiple columns for multiple sequencing runs)
* `--mutation_fitness` or `-s`: a .csv file, with
  + 1st column: total beneficial mutation rate, Ub
  + 2nd column: bin edges of the arbitrary DFE
  + 3rd column: counts frequency in each bin of the 2nd column
* `--maximum_mutation_number` or `-max_mut_num`: maximum number of mutations allowed in each single cell (`default: 1`)
* `--t_pregrowth` or `-t_pre`: number of generations in pre-growth (`default: 16`)
* `--cell_num_average_bottleneck` or `-n_b`: average number of cells per barcode transferred at each bottleneck' (`default: 100`)
* `--c` or `-c`: half of variance introduced by cell growth and cell transfer' (`default: 1`)
* `--dna_copies` or `-d`: average genome template copies per barcode in PCR' (`default: 500`)
* `--pcr_cycles` or `-p`: number of cycles in PCR' (`default: 25`)
* `--output_filename` or `-o`: prefix of output files' (`default: output`)

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
python fitmutsimu_run_v1.py --help
```

##### Examples
```
python fitmutsimu_run_v1.py -l 10000 -t simu_input_time_points.csv -s simu_input_mutation_fitness.csv -o test
```    


#### 3.2. Mutations Identification
`fitmut2_run_v1.py` identifies adaptive mutations in barcoded evolution experiments from read-count time-series data, and estimates their fitness effects and establishment times. 

##### Options
* `--input` or `-i`: a .csv file, with each column being the read number per barcode at each sequenced time-point
* `--t_seq` or `-t`: a .csv file, with
  + 1st column: sequenced time-points evaluated in number of generations
  + 2nd column: total effective number of cells of the population for each sequenced time-point
* `--mutation_rate` or `-u`: total beneficial mutation rate (`default: 1e-5`)
* `--delta_t` or `-dt`: number of generations between bottlenecks (`default: 8`)
* `--c` or `-c`: half of variance introduced by cell growth and cell transfer' (`default: 1`)
* `--maximum_iteration_number` or `-n`: maximum number of iterations (`default: 50`)
* `--opt_algorithm` or `-a`: optmization algorithm (nelder_mead or differential_evolution) (`default: nelder_mead`)
* `--parallelize` or `-p`: whether to use multiprocess module to parallelize inference across lineages (`default: True`)
* `--save_steps` or `-s`: whether to output files in intermediate step of iterations (`default: False`)
* `--output_filename` or `-o`: prefix of output files' (`default: output`)

##### Outputs
* `output_MutSeq_Result.csv`: a .csv file, with
  + 1st column of .csv: estimated fitness effect of each lineage
  + 2nd column of .csv: estimated establishment time of each lineage
  + 3rd column of .csv: theoretical estimation error for fitness effect
  + 4th column of .csv: theoretical estimation error for establishment time
  + 5th column of .csv: probability of each lineage containing an adaptive mutation
  + 6th column of .csv: estimated mean fitness per sequenced time point
  + 7th column of .csv: estimated kappa per sequenced time point
  + 8th column of .csv: estimated fraction of mutant cells of the population per sequenced time point
* `output_Mean_fitness_Result.csv`: estimated mean fitness at each iteration
* `output_Cell_Number_Mutant_Estimated.csv`: estimated effective number of mutant cells per barcode for each time point
* `output_Cell_Number.csv`: effective cell number per barcode for each time point 

##### For Help
```
python fitmut2_run_v1.py --help
```  

##### Examples
```
python fitmut2_run_v1.py -i simu_test_EvoSimulation_Read_Number.csv -t fitmut_input_time_points.csv -o test
```
