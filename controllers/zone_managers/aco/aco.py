import random
import numpy as np

# ACO hyper-parameters
NUM_ANTS = 10
NUM_ITERATIONS = 100
ALPHA = 1.0  # Pheromone influence
BETA = 2.0  # Heuristic influence
EVAPORATION_RATE = 0.1
Q = 100.0  # Pheromone increase factor

# heuristic hyper-parameters
DISTANCE_FACTOR = 1.0 / 50.0
EXEC_TIME_FACTOR = 1.0


class ACO:
    # Task execution options: local (0), fog nodes (1...n), cloud (n+1)

    def __init__(self, distances: list[float], exec_times: list[float]):
        # print('started ACO with:')
        # print('DST', distances)
        # print('EXE', exec_times)
        if len(distances) != len(exec_times):
            raise ValueError("distances and exec_times must have the same length.")
        self.num_options = len(distances)
        self.num_fog_nodes = self.num_options - 2

        self.distances = distances
        self.exec_times = exec_times

        # Initialize pheromone levels equally:
        self.pheromone = np.ones(self.num_options)

    def ant_decision(self):
        heuristics = np.array([self.heuristic(i) for i in range(self.num_options)])
        probs = (self.pheromone ** ALPHA) * (heuristics ** BETA)
        probs /= np.sum(probs)
        return self.choose_option(probs)

    def get_cost(self, option_index):
        return DISTANCE_FACTOR * self.distances[option_index] + EXEC_TIME_FACTOR * self.exec_times[option_index]

    # Heuristic information (e.g., inverse of cost)
    def heuristic(self, option_index):
        return 1.0 / self.get_cost(option_index)

    def evaluate_solution(self, option_index):
        # Define cost function: lower is better
        return self.get_cost(option_index)

    def choose_option(self, probabilities):
        r = random.random()
        total = 0
        for idx, p in enumerate(probabilities):
            total += p
            if r <= total:
                return idx
        return len(probabilities) - 1  # fallback

    def run(self):
        # ACO main loop
        for iteration in range(NUM_ITERATIONS):
            all_choices = []
            all_costs = []

            for ant in range(NUM_ANTS):
                choice = self.ant_decision()
                cost = self.evaluate_solution(choice)
                all_choices.append(choice)
                all_costs.append(cost)

            # Evaporate pheromone
            self.pheromone *= (1 - EVAPORATION_RATE)

            # Update pheromones based on ant performance
            for choice, cost in zip(all_choices, all_costs):
                self.pheromone[choice] += Q / cost

            # best_idx = np.argmin(all_costs)
            # best_choice = all_choices[best_idx]
            # print(f"Iteration {iteration}: Best Option = {best_choice}, Cost = {all_costs[best_idx]:.2f}")

        # Final Decision
        final_decision = np.argmax(self.pheromone)
        # options = ['local'] + [f'fog-{i + 1}' for i in range(self.num_fog_nodes)] + ['cloud']
        # print(f"Final best execution location: {options[final_decision]}\n===================")
        return final_decision


if __name__ == '__main__':
    print(ACO(
        [1, 2, 3, 1.01, 5],
        [1, 2, 3, 1, 5],
    ).run())
