# search/genetic_ops.py
import random
from models.operations import OP_MAPPING


class GeneticOperators:
    def __init__(self, crossover_rate, mutation_rate, num_ops):
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_ops = num_ops

    def _find_idx(self, target_attrs):
        """İstenilen DNA özelliklerine sahip genin indeksini bulur."""
        for idx, attrs in OP_MAPPING.items():
            if attrs == target_attrs:
                return idx
        return random.randint(0, self.num_ops - 1)

    def mutate_gene(self, gene_idx):
        """Bir genin sadece tek bir eksenini (örneğin sadece kernel'i) değiştirir."""
        attrs = OP_MAPPING[gene_idx].copy()

        # Identity veya RepConv ise rastgele başka bir şeye dönüşsün
        if attrs['type'] in ['Identity', 'RepConv']:
            return random.randint(0, self.num_ops - 1)

        # Sadece 1 özelliği seç ve mutasyona uğrat
        axis = random.choice(['type', 'k', 'e', 'se'])

        if axis == 'type':
            types = ['MBConv', 'FusedMBConv', 'CSPMBConv']
            types.remove(attrs['type'])
            attrs['type'] = random.choice(types)
        elif axis == 'k':
            attrs['k'] = 5 if attrs['k'] == 3 else 3
        elif axis == 'e':
            exps = [1, 3, 6]
            exps.remove(attrs['e'])
            attrs['e'] = random.choice(exps)
        elif axis == 'se':
            attrs['se'] = 1 if attrs['se'] == 0 else 0

        return self._find_idx(attrs)

    def generate_offspring(self, p1, p2):
        # Uniform crossover, applied with probability crossover_rate;
        # otherwise the child is a clone of p1 (then mutated).
        if random.random() < self.crossover_rate:
            child = [g1 if random.random() < 0.5 else g2 for g1, g2 in zip(p1, p2)]
        else:
            child = list(p1)

        return [self.mutate_gene(g) if random.random() < self.mutation_rate else g for g in child]