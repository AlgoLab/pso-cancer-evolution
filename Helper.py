class Helper(object):
    def __init__(self, matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, max_deletions, max_time):
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
        self.avg_dist = 0
