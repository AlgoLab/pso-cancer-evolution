class Helper(object):
    def __init__(self, matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, w, c1, c2, max_deletions, max_time):
        self.matrix = matrix
        self.mutation_number = mutation_number
        self.mutation_names = mutation_names
        self.cells = cells
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.max_deletions = max_deletions
        self.max_time = max_time
        self.best_particle = None

        # #riscalo w,c1,c2 che la loro somma faccia 1
        somma = sum([w, c1, c2])
        self.w = w/somma
        self.c1 = c1/somma
        self.c2 = c2/somma

        self.avg_dist = 0
