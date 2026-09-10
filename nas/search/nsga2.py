# search/nsga2.py
import os
import json
import random
from search.genetic_ops import GeneticOperators
from models.operations import OPS
from utils.visualize import plot_pareto_front, plot_evolution


class NSGA2:
    def __init__(self, config, evaluator, lut_manager):
        self.config = config
        self.evaluator = evaluator
        self.lut = lut_manager

        # FIX (critical): this was hardcoded to 4 while OPS has 5 entries.
        # GeneticOperators was already built with num_ops=len(OPS)=5, so
        # mutation could introduce op index 4 into a chromosome even though
        # _init_population() (which used the old hardcoded 4) - and, more
        # importantly, NASEngine's fair-path training loop - never touched
        # that op's weights. Any chromosome containing an untrained op 4
        # would get evaluated with garbage weights, silently corrupting its
        # fitness score. Every module must agree on the same num_ops.
        self.num_ops = len(OPS)

        self.cache = {}

        self.genetic_ops = GeneticOperators(
            crossover_rate=config.crossover_rate,
            mutation_rate=config.mutation_rate,
            num_ops=self.num_ops
        )

    def calculate_crowding_distance(self, front, fitness):
        """Distance from an individual's nearest neighbours in objective
        space, within a single Pareto front. Boundary individuals (best on
        either objective) get infinite distance and are always preserved."""
        distance = {i: 0.0 for i in front}

        for obj_idx in range(len(fitness[front[0]])):
            front_sorted = sorted(front, key=lambda i: fitness[i][obj_idx])
            distance[front_sorted[0]] = float('inf')
            distance[front_sorted[-1]] = float('inf')

            obj_vals = [fitness[i][obj_idx] for i in front_sorted]
            obj_range = max(obj_vals) - min(obj_vals) or 1e-9

            for k in range(1, len(front_sorted) - 1):
                distance[front_sorted[k]] += (obj_vals[k + 1] - obj_vals[k - 1]) / obj_range

        return distance

    def tournament_select(self, population, ranks, crowds):
        """Binary tournament: lower (better) rank wins; ties broken by
        higher crowding distance (more diverse individual)."""
        a, b = random.sample(range(len(population)), 2)
        if ranks[a] != ranks[b]:
            winner = a if ranks[a] < ranks[b] else b
        else:
            winner = a if crowds[a] > crowds[b] else b
        return population[winner]

    def _init_population(self):
        return [[random.randint(0, self.num_ops - 1) for _ in range(self.config.num_layers)]
                for _ in range(self.config.pop_size)]

    def _evaluate_population(self, population):
        """Evaluates fitness for each chromosome, caching results by
        architecture so identical chromosomes (common after crossover/
        elitism) are never re-evaluated."""
        fitness = []
        for chrom in population:
            chrom_tuple = tuple(chrom)
            if chrom_tuple in self.cache:
                fitness.append(self.cache[chrom_tuple])
            else:
                f1 = self.evaluator.evaluate_subnet(chrom)
                lat = self.lut.get_subnet_latency(chrom)
                res = (f1, lat)
                self.cache[chrom_tuple] = res
                fitness.append(res)
        return fitness

    def dominates(self, obj1, obj2):
        """obj = (f1, latency): f1 higher-is-better, latency lower-is-better."""
        f1_1, lat_1 = obj1
        f1_2, lat_2 = obj2
        return (f1_1 >= f1_2 and lat_1 <= lat_2) and (f1_1 > f1_2 or lat_1 < lat_2)

    def non_dominated_sorting(self, population, fitness):
        """Fast non-dominated sort (Deb et al., 2002). Returns fronts from
        best (index 0 = Pareto front) to worst."""
        S = [[] for _ in range(len(population))]
        fronts = [[]]
        n = [0 for _ in range(len(population))]

        for p in range(len(population)):
            for q in range(len(population)):
                if self.dominates(fitness[p], fitness[q]):
                    S[p].append(q)
                elif self.dominates(fitness[q], fitness[p]):
                    n[p] += 1
            if n[p] == 0:
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return fronts[:-1]

    def _rank_and_crowd(self, fronts, fitness):
        ranks, crowds = {}, {}
        for r_idx, front in enumerate(fronts):
            cd = self.calculate_crowding_distance(front, fitness)
            for idx in front:
                ranks[idx] = r_idx
                crowds[idx] = cd[idx]
        return ranks, crowds

    def _environmental_selection(self, population, fitness, size):
        """Standard NSGA-II (mu+lambda) survival: fill by front, truncate the
        last front by crowding distance."""
        fronts = self.non_dominated_sorting(population, fitness)
        next_pop, next_fit = [], []
        for front in fronts:
            if len(next_pop) + len(front) <= size:
                chosen = front
            else:
                cd = self.calculate_crowding_distance(front, fitness)
                chosen = sorted(front, key=lambda i: cd[i], reverse=True)[: size - len(next_pop)]
            next_pop.extend(population[i] for i in chosen)
            next_fit.extend(fitness[i] for i in chosen)
            if len(next_pop) >= size:
                break
        return next_pop, next_fit

    def run(self):
        population = self._init_population()
        fitness = self._evaluate_population(population)

        history = []
        gen_dir = os.path.join(self.config.save_dir, "generations")
        plots_dir = os.path.join(self.config.save_dir, "plots")
        os.makedirs(gen_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        for gen in range(self.config.num_generations):
            fronts = self.non_dominated_sorting(population, fitness)
            ranks, crowds = self._rank_and_crowd(fronts, fitness)

            pareto_idx = fronts[0]
            gen_record = {
                "generation": gen + 1,
                "pareto_front": [population[i] for i in pareto_idx],
                "metrics": [fitness[i] for i in pareto_idx],
            }
            history.append(gen_record)
            with open(os.path.join(gen_dir, f"gen_{gen + 1}.json"), "w") as f:
                json.dump(gen_record, f, indent=2)
            plot_pareto_front(gen + 1, fitness, plots_dir)

            best_f1 = max(f for f, _ in fitness)
            min_lat = min(l for _, l in fitness)
            print(f"Generation {gen + 1}/{self.config.num_generations} | Pareto size: {len(pareto_idx)} | "
                  f"best F1: {best_f1:.4f} | min latency: {min_lat:.2f}ms | evaluated: {len(self.cache)}")

            # lambda = pop_size offspring, evaluated once (cache dedups repeats)
            offspring = []
            while len(offspring) < self.config.pop_size:
                p1 = self.tournament_select(population, ranks, crowds)
                p2 = self.tournament_select(population, ranks, crowds)
                offspring.append(self.genetic_ops.generate_offspring(p1, p2))
            offspring_fitness = self._evaluate_population(offspring)

            # (mu + lambda): parents compete with children for survival
            population, fitness = self._environmental_selection(
                population + offspring, fitness + offspring_fitness, self.config.pop_size
            )

        plot_evolution(history, plots_dir)
        final_fronts = self.non_dominated_sorting(population, fitness)
        return final_fronts, population, fitness