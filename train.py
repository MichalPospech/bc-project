# training settings
""" roboschool envs available
robo_pendulum
robo_double_pendulum
robo_reacher
robo_flagrun

robo_ant
robo_reacher
robo_hopper
robo_walker
robo_humanoid
"""

from mpi4py import MPI
import numpy as np
import json
import os
import subprocess
import sys
import config
from model import make_model, simulate
from es import CMAES, NSES, SimpleGA, OpenES, PEPG, NSRAES, NSRES
import argparse
import time

### MPI related code
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_worker = comm.Get_size() - 1

PRECISION = 10000


class Experiment(object):
    def __init__(
        self,
        gamename,
        algorithm,
        num_episode,
        eval_steps,
        num_worker_trial,
        cap_time,
        retrain,
        seed_start,
        batch_mode,
    ):
        algorithm_name = algorithm["name"]
        self.game = config.games[gamename]
        self.gamename = gamename
        self.population = num_worker * num_worker_trial
        self.retrain_mode = retrain
        self.cap_time_mode = cap_time
        self.seed_start = seed_start
        self.model = make_model(self.game)
        self.optimizer = Experiment.get_algorithm(
            self.model.param_count, self.population, **algorithm
        )
        self.num_params = self.model.param_count
        self.num_worker_trial = num_worker_trial
        self.num_episode = num_episode
        self.log_filebase = (
            "log/"
            + gamename
            + "."
            + algorithm_name
            + "."
            + str(self.num_episode)
            + "."
            + str(self.population)
            + "."
            + str(int(time.time()))
        )
        self.batch_mode = batch_mode
        self.eval_steps = eval_steps

    def get_solution_packet_size(self):
        return (5 + self.num_params) * self.num_worker_trial

    def get_result_packet_size(self):
        return (4 + self.game.input_size) * self.num_worker_trial

    @staticmethod
    def get_algorithm(num_params, population, algorithm_params):
        optimizer_name = algorithm_params["name"]
        algorithm_params.pop("name")
        es = None
        if optimizer_name == "ga":
            es = SimpleGA(num_params, popsize=population, **algorithm_params)
        elif optimizer_name == "cma":
            es = CMAES(num_params, popsize=population, **algorithm_params)
        elif optimizer_name == "pepg":
            es = PEPG(num_params, popsize=population, **algorithm_params)
        elif optimizer_name == "nses":
            es = NSES(num_params, popsize=population, **algorithm_params)
        elif optimizer_name == "nsres":
            es = NSRES(num_params, popsize=population, **algorithm_params)
        elif optimizer_name == "nsraes":
            es = NSRAES(num_params, popsize=population, **algorithm_params)
        elif optimizer_name == "openes":
            es = OpenES(num_params, popsize=population, **algorithm_params)
        else:
            raise ValueError(f"Unknown optimizer name {optimizer_name}")
        return es


def sprint(*args):
    print(*args)  # if python3, can do print(*args)
    sys.stdout.flush()


class OldSeeder:
    def __init__(self, init_seed=0):
        self._seed = init_seed

    def next_seed(self):
        result = self._seed
        self._seed += 1
        return result

    def next_batch(self, batch_size):
        result = np.arange(self._seed, self._seed + batch_size).tolist()
        self._seed += batch_size
        return result


class Seeder:
    def __init__(self, init_seed=0):
        np.random.seed(init_seed)
        self.limit = np.int32(2 ** 31 - 1)

    def next_seed(self):
        result = np.random.randint(self.limit)
        return result

    def next_batch(self, batch_size):
        result = np.random.randint(self.limit, size=batch_size).tolist()
        return result


class Communicator:
    def __init__(
        self,
        precision,
        solution_packet_size,
        result_packet_size,
        num_worker_trial,
        final_state_size,
    ):
        self.precision = precision
        self.solution_packet_size = solution_packet_size
        self.result_packet_size = result_packet_size
        self.num_worker_trial = num_worker_trial
        self.final_state_size = final_state_size

    def encode_solution_packets(self, seeds, solutions, train_mode=1, max_len=-1):
        n = len(seeds)
        result = []
        worker_num = 0
        for i in range(n):
            worker_num = int(i / self.num_worker_trial) + 1
            result.append([worker_num, i, seeds[i], train_mode, max_len])
            result.append(np.round(np.array(solutions[i]) * self.precision, 0))
        result = np.concatenate(result).astype(np.int32)
        result = np.split(result, num_worker)
        return result

    def decode_solution_packet(self, packet):
        packets = np.split(packet, self.num_worker_trial)
        result = []
        for p in packets:
            result.append(
                [p[0], p[1], p[2], p[3], p[4], p[5:].astype(np.float) / self.precision]
            )
        return result

    def encode_result_packet(self, results):
        r = np.array(results)
        r[:, 2:] *= self.precision
        return r.flatten().astype(np.int32)

    def decode_result_packet(self, packet):
        r = packet.reshape(self.num_worker_trial, 4 + self.final_state_size)
        workers = r[:, 0].tolist()
        jobs = r[:, 1].tolist()
        fits = r[:, 2].astype(np.float) / self.precision
        fits = fits.tolist()
        times = r[:, 3].astype(np.float) / self.precision
        times = times.tolist()
        positions = r[:, 4:].astype(np.float) / self.precision
        positions = positions.tolist()

        result = []
        n = len(jobs)
        for i in range(n):
            result.append([workers[i], jobs[i], fits[i], times[i], *positions[i]])
        return result

    def recive_solution_packet(self):
        packet = np.empty(self.solution_packet_size, dtype=np.int32)
        comm.Recv(packet, source=0)
        assert len(packet) == self.solution_packet_size
        return self.decode_solution_packet(packet)

    def send_results_packet(self, results):
        result_packet = self.encode_result_packet(results)
        assert len(result_packet) == self.result_packet_size
        comm.Send(result_packet, dest=0)

    def send_packets_to_slaves(self, packet_list):
        assert len(packet_list) == num_worker
        for i in range(1, num_worker + 1):
            packet = packet_list[i - 1]
            assert len(packet) == self.solution_packet_size
            comm.Send(packet, dest=i)

    def receive_packets_from_slaves(self, population):
        result_packet = np.empty(self.result_packet_size, dtype=np.int32)
        reward_list_total = np.zeros((population, 2))
        check_results = np.ones(population, dtype=np.int)
        for i in range(1, num_worker + 1):
            comm.Recv(result_packet, source=i)
            results = self.decode_result_packet(result_packet)
            for result in results:
                worker_id = int(result[0])
                possible_error = "work_id = " + str(worker_id) + " source = " + str(i)
                assert worker_id == i, possible_error
                idx = int(result[1])
                reward_list_total[idx, 0] = result[2]
                reward_list_total[idx, 1] = result[3]
                check_results[idx] = 0
        check_sum = check_results.sum()
        assert check_sum == 0, check_sum
        return reward_list_total


def worker(experiment, weights, seed, train_mode_int=1, max_len=-1):

    train_mode = train_mode_int == 1
    experiment.model.set_model_params(weights)
    reward_list, t_list, end_state = simulate(
        experiment.model,
        train_mode=train_mode,
        render_mode=False,
        num_episode=experiment.num_episode,
        seed=seed,
        max_len=max_len,
    )
    if experiment.batch_mode == "min":
        reward = np.min(reward_list)
    else:
        reward = np.mean(reward_list)
    t = np.mean(t_list)
    return reward, t, end_state


def slave(experiment, communicator):
    experiment.model.make_env()
    while 1:
        solutions = communicator.recive_solution_packet()
        results = []
        for solution in solutions:
            worker_id, jobidx, seed, train_mode, max_len, weights = solution
            assert train_mode == 1 or train_mode == 0, str(train_mode)
            worker_id = int(worker_id)
            possible_error = "work_id = " + str(worker_id) + " rank = " + str(rank)
            assert worker_id == rank, possible_error
            jobidx = int(jobidx)
            seed = int(seed)
            fitness, timesteps, end_state = worker(
                experiment, weights, seed, train_mode, max_len
            )
            results.append([worker_id, jobidx, fitness, timesteps, end_state])
        communicator.send_results_packet(results)


def evaluate_batch(optimizer, model_params, communicator, max_len=-1):
    # duplicate model_params
    solutions = []
    for _ in range(optimizer.popsize):
        solutions.append(np.copy(model_params))

    seeds = np.arange(optimizer.popsize)

    packet_list = communicator.encode_solution_packets(
        seeds, solutions, train_mode=0, max_len=max_len
    )

    communicator.send_packets_to_slaves(packet_list)
    reward_list_total = communicator.receive_packets_from_slaves(optimizer.popsize)

    reward_list = reward_list_total[:, 0]  # get rewards
    return np.mean(reward_list)


def master(experiment, communicator):

    start_time = int(time.time())
    sprint("training", experiment.gamename)
    sprint("population", experiment.optimizer.popsize)
    sprint("num_worker", num_worker)
    sprint("num_worker_trial", experiment.num_worker_trial)
    sys.stdout.flush()

    seeder = Seeder(experiment.seed_start)

    filename = experiment.log_filebase + ".json"
    filename_log = experiment.log_filebase + ".log.json"
    filename_hist = experiment.log_filebase + ".hist.json"
    filename_best = experiment.log_filebase + ".best.json"

    experiment.model.make_env()

    def evaluate_locally(weights, experiment, seed, max_len):
        experiment.model.set_model_params(weights)
        reward_list, _, end_states = simulate(
            experiment.model,
            train_mode=False,
            render_mode=False,
            num_episode=experiment.num_episode,
            seed=seed,
            max_len=max_len,
        )
        reward = reward_list[0]
        end_state = end_states[0]
        return reward, end_state

    def evaluate_initials(solutions):
        if experiment.antitethic:
            seeds = seeder.next_batch(int(experiment.optimizer.popsize / 2))
            seeds = seeds + seeds
        else:
            seeds = seeder.next_batch(experiment.optimizer.popsize)

        chars = []
        rewards = []
        for i in range(solutions.dim[0]):
            r, c = evaluate_locally(solutions[i, :], experiment, seeds[i], max_len)
            rewards.append(r)
            chars.append(c)
        return rewards, chars

    experiment.optimizer.init(evaluate_initials)
    t = 0

    history = []
    eval_log = []
    best_reward_eval = 0
    best_model_params_eval = None

    max_len = -1  # max time steps (-1 means ignore)

    while True:
        t += 1
        es = experiment.optimizer
        solutions = experiment.optimizer.ask()

        if experiment.antitethic:
            seeds = seeder.next_batch(int(experiment.optimizer.popsize / 2))
            seeds = seeds + seeds
        else:
            seeds = seeder.next_batch(experiment.optimizer.popsize)

        packet_list = communicator.encode_solution_packets(
            seeds, solutions, max_len=max_len
        )

        communicator.send_packets_to_slaves(packet_list)
        reward_list_total = communicator.receive_packets_from_slaves(
            experiment.optimizer.popsize
        )

        reward_list = reward_list_total[:, 0]  # get rewards

        mean_time_step = (
            int(np.mean(reward_list_total[:, 1]) * 100) / 100.0
        )  # get average time step
        max_time_step = (
            int(np.max(reward_list_total[:, 1]) * 100) / 100.0
        )  # get average time step
        avg_reward = int(np.mean(reward_list) * 100) / 100.0  # get average time step
        std_reward = int(np.std(reward_list) * 100) / 100.0  # get average time step
        chars = reward_list_total[:, 4:]

        experiment.optimizer.tell(
            reward_list,
            chars,
            lambda weights: evaluate_locally(
                weights, experiment, seeder.next_seed(), max_len
            ),
        )

        es_solution = experiment.optimizer.result()
        model_params = es_solution[0]  # best historical solution
        experiment.model.set_model_params(np.array(model_params).round(4))

        r_max = int(np.max(reward_list) * 100) / 100.0
        r_min = int(np.min(reward_list) * 100) / 100.0

        curr_time = int(time.time()) - start_time

        h = (
            t,
            curr_time,
            avg_reward,
            r_min,
            r_max,
            std_reward,
            int(es.rms_stdev() * 100000) / 100000.0,
            mean_time_step + 1.0,
            int(max_time_step) + 1,
        )

        if experiment.cap_time_mode:
            max_len = 2 * int(mean_time_step + 1.0)
        else:
            max_len = -1

        history.append(h)

        with open(filename, "wt") as out:
            json.dump(
                [np.array(experiment.optimizer.current_param()).round(4).tolist()],
                out,
                sort_keys=True,
                indent=2,
                separators=(",", ": "),
            )

        with open(filename_hist, "wt") as out:
            json.dump(history, out, sort_keys=False, indent=0, separators=(",", ":"))

        sprint(experiment.gamename, h)

        if t == 1:
            best_reward_eval = avg_reward
        if t % experiment.eval_steps == 0:  # evaluate on actual task at hand

            prev_best_reward_eval = best_reward_eval
            model_params_quantized = np.array(es.current_param()).round(4)
            reward_eval = evaluate_batch(
                experiment.optimizer, model_params_quantized, communicator, max_len=-1
            )
            model_params_quantized = model_params_quantized.tolist()
            improvement = reward_eval - best_reward_eval
            eval_log.append([t, reward_eval, model_params_quantized])
            with open(filename_log, "wt") as out:
                json.dump(eval_log, out)
            if len(eval_log) == 1 or reward_eval > best_reward_eval:
                best_reward_eval = reward_eval
                best_model_params_eval = model_params_quantized
            else:
                if experiment.retrain_mode:
                    sprint(
                        "reset to previous best params, where best_reward_eval =",
                        best_reward_eval,
                    )
                    es.set_mu(best_model_params_eval)
            with open(filename_best, "wt") as out:
                json.dump(
                    [best_model_params_eval, best_reward_eval],
                    out,
                    sort_keys=True,
                    indent=0,
                    separators=(",", ": "),
                )
            sprint(
                "improvement",
                t,
                improvement,
                "curr",
                reward_eval,
                "prev",
                prev_best_reward_eval,
                "best",
                best_reward_eval,
            )


# TODO Assert parameters (num episodes = 1 for novelty, etc.)
# TODO Config from JSON
def main(params):
    num_worker_trial = params["num_worker_trial"]
    params.pop("num_worker_trial")
    experiment = Experiment(**params)
    communicator = Communicator(
        10000,
        (5 + experiment.model.param_count) * num_worker_trial,
        4 * num_worker_trial,
        num_worker_trial,
        experiment.game.input_size,
    )

    sprint("process", rank, "out of total ", comm.Get_size(), "started")
    if rank == 0:
        master(experiment, communicator)
    else:
        slave(experiment, communicator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train policy on OpenAI Gym environment "
            "using pepg, ses, openes, ga, cma, nses, nsres, nsraes"
        )
    )
    parser.add_argument(
        "filename", "-f", type=str, help="robo_pendulum, robo_ant, robo_humanoid, etc."
    )
    args = parser.parse_args()
    parameters = None

    with open(args.filename, "r", encoding="utf-8") as json_file:
        parameters = json.load(json_file)
    main(parameters)