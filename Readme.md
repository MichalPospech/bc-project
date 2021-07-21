# Installation

To download simply clone the repository using `git clone --recursive https://github.com/MichalPospech/bc-project.git` to include the `estool` library included. Then simply install the local dependency using `pip install -e ./libs/estool`. 

An installation of MPI (for mpi4py) is also needed, usually possible to install using your distribution's package manager.

On Windows, it is easiest to install mpi4py as follows:

- Download and install mpi_x64.Msi from the HPC Pack 2012 MS-MPI Redistributable Package
- Install a recent Visual Studio version with C++ compiler
- Open a command prompt

```
git clone https://github.com/mpi4py/mpi4py
cd mpi4py
python setup.py install
```

# Running

The program is run using MPI, therefore the running command is a bit more complex.
`mpiexec -n NUM_CPU python -m mpi4py train.py -f CONFIG_FILE`

where `NUM_CPU` must be at least 2 and at most number of physical cores and `CONFIG_FILE` a path to config file with format described below

# Configuration file format
Configuration file is in JSON format with specification and example below

## Main
- `num_worker_trial` - number of trials per agent
- `gamename` - name of game, `slimevolley` or `cartpole_swingup`
- `algorithm` - Algorithm parameters, see algorithm section
- `num_episode` - number of episodes used to evaluate each solution
- `batch_mode` - how are the `num_episode` values for each solution aggregated, `min` or `mean`
- `eval_steps` - how often is current solution evaluated
- `cap_time` - limit on number of steps per episode (-1 for unlimited, default)
- `seed_start` - starting seed for RNG
- `antitethic` - whether antitethic sampling should be used (default is `True`)


## Algorithm
- `name` - name of algorithm used, `cmaes`, `ga`, `pepg`,`openes`,`nses`,`nsres` or `nsraes`
### CMA-ES
- `sigma_init` - initial step size as per CMA-ES specification 
- `weight_decay` - weight decay subtracted from rewards

### Simple genetic algorithm
- `sigma_init` - initial sigma for distribution
- `sigma_decay` 
- `sigma_limit`
- `elite_ratio`
- `forget_best` - update the individual all the time, not only upon improvement
- `weight_decay` - weight decay subtracted from rewards

### Open AI ES
- `optimizer`
- `sigma_init`
- `sigma_decay`
- `sigma_limit`
- `forget_best` - update the individual all the time, not only upon improvement
- `weight_decay`
- `rank_fitness`

### PEPG
- `sigma_init`
- `sigma_decay`
- `sigma_limit`
- `sigma_max_change`
- `learning_rate`
- `learning_rate_decay`
- `learning_rate_limit`
- `elite_ratio`
- `average_baseline`
- `weight_decay`
- `rank_fitness`
- `forget_best`

### NSES
- `optimizer`
- `sigma`
- `metapopulation_size`
- `k`
### NSRES
- `optimizer`
- `sigma`
- `metapopulation_size`
- `k`
- `weight`
### NSRAES
- `optimizer`
- `sigma`
- `metapopulation_size`
- `k`
- `init_weight`
- `weight_change`
- `weight_change_threshold`

## Optimizers
- `name` - name of used optimizer, `sgd`, `adam` or `sgdm`

All the parameters should be self-explanatory as per the algorithm definitions

### SGD 
- `stepsize`

### SGD with momentum
- `stepsize`
- `momentum`

### Adam
- `stepsize`
- `beta1`
- `beta2`


