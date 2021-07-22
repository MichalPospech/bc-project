# Installation

To download simply clone the repository using `git clone --recursive https://github.com/MichalPospech/bc-project.git` to include the `estool` library included. Then simply install the local dependency using `pip install -e ./libs/estool`. 
Also a `log` folder must be created.

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

# Logfiles
There are 4 log files
 - hist contains data from training (timestep, total time, avg, std, min and max for both reward and novelty)
 - best contains currently best parameters with reward
 - log contains evaluation data (timestep, reward and parameters)
 - no extenstion contains current parameters 

# Configuration file format
Configuration file is in JSON format with specification below, examples in `examples` directory

## Main
- `num_worker_trial` - number of individuals PER WORKER
- `gamename` - name of game, `slimevolley` or `cartpole_swingup`
- `algorithm` - Algorithm parameters, see algorithm section
- `num_episode` - number of episodes used to evaluate each solution
- `batch_mode` - how are the `num_episode` values for each solution aggregated, `min` or `mean`
- `eval_steps` - how often is current solution evaluated
- `cap_time` - limit on number of steps per episode (-1 for unlimited, default)
- `seed_start` - starting seed for RNG
- `antitethic` - whether antitethic sampling should be used (default is `True`)
- `identifier` - identifier to identify logfile by


## Algorithm
- `name` - name of algorithm used, `cmaes`, `ga`, `pepg`,`openes`,`nses`,`nsres` or `nsraes`

If some parameter is undescribed, it is because it is usual naming for the algorithm
### CMA-ES
- `sigma_init` - initial step size as per CMA-ES specification 
- `weight_decay` - weight decay subtracted from rewards

### Simple genetic algorithm
- `sigma_init` - initial sigma for distribution
- `sigma_decay` - rate of decay
- `sigma_limit` - limit for decay
- `elite_ratio`
- `forget_best` - update the individual all the time, not only upon improvement
- `weight_decay` - weight decay subtracted from rewards

### Open AI ES
- `optimizer` - see optimizer section
- `sigma_init` - initial sigma for distribution
- `sigma_decay` - rate of decay
- `sigma_limit` - limit for decay
- `forget_best` - update the individual all the time, not only upon improvement
- `weight_decay` - weight decay subtracted from rewards
- `rank_fitness` - use rank-normalised fitness

### PEPG
- `sigma_init` - initial sigma for distribution
- `sigma_decay` - rate of decay
- `sigma_limit` - limit for decay
- `sigma_max_change`
- `learning_rate` - initial learning rate
- `learning_rate_decay` - rate of learning rate decay
- `learning_rate_limit` - limit for learning rate
- `weight_decay` - weight decay subtracted from rewards
- `rank_fitness` - use rank-normalised fitness

### NSES
- `optimizer` - see optimizer section
- `sigma`
- `metapopulation_size`
- `k`
### NSRES
- `optimizer` - see optimizer section
- `sigma`
- `metapopulation_size`
- `k`
- `weight` - ratio of fitness and novelty
### NSRAES 
- `optimizer` - see optimizer section
- `sigma`
- `metapopulation_size`
- `k`
- `init_weight` - initial ratio of fitness and novelty
- `weight_change` - how much does the ratio change 
- `weight_change_threshold` - how often does the ratio change

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


