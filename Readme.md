# Configuration file format
## Main
- `num_worker_trial` - number of trials per agent
- `gamename`
- `algorithm` - Algorithm parameters, see algorithm secion
- `num_episode`
- `eval_steps`
- `cap_time`
- `retrain`
- `seed_start`
- `batch_mode`
- `antitethic`


## Algorithm
- `name`
### CMA-ES
- `sigma_init`
- `weight_decay`

### Simple genetic algorithm
- `sigma_init`
- `sigma_decay`
- `sigma_limit`
- `elite_ratio`
- `forget_best`
- `weight_decay`

### Open AI ES
- `optimizer`
- `sigma_init`
- `sigma_decay`
- `sigma_limit`
- `elite_ratio`
- `forget_best`
- `weight_decay`
- `learning_rate`
- `learning_rate_decay`
- `learning_rate_limit`
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
- `name`

### SGD 
- `stepsize`
### SGD with momentum
- `stepsize`
- `momentum`

### Adam
- `stepsize`
- `beta1`
- `beta2`


