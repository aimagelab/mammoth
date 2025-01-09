import numpy as np


class BetaMixture1D(object):
    """
    Fits a Beta-Mixture model to the data.

    This code is based on the https://github.com/PaulAlbert31/LabelNoiseCorrection/blob/master/utils.py
    """

    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        try:
            import scipy.stats
        except ImportError:
            raise ImportError("scipy is required for BetaMixture1D. Install it with `pip install scipy`.")

        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    @staticmethod
    def fit_beta_weighted(x, w):
        def weighted_mean(x, w):
            return np.sum(w * x) / np.sum(w)

        x_bar = weighted_mean(x, w)
        s2 = weighted_mean((x - x_bar)**2, w)
        # s2 += np.finfo(s2.dtype).eps
        alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
        beta = alpha * (1 - x_bar) / x_bar
        return alpha, beta

    @staticmethod
    def outlier_remove(x):
        # outliers detection
        max_perc = np.percentile(x, 95)
        min_perc = np.percentile(x, 5)
        x = x[(x <= max_perc) & (x >= min_perc)]
        x_max = max_perc
        x_min = min_perc + 10e-6
        return x, x_min, x_max

    @staticmethod
    def normalize(x, x_min, x_max):
        # normalized the centrality for bmm
        x = (x - x_min) / (x_max - x_min + 1e-6)
        x[x >= 1] = 1 - 10e-4
        x[x <= 0] = 10e-4
        return x

    def likelihood(self, x, y):
        import scipy.stats as stats

        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        wl = self.weighted_likelihood(x, y)
        p = self.probability(x)
        pos = wl / (p + self.eps_nan)

        wl_inf = np.isinf(wl)
        p_inf = np.isinf(p)

        # inf / inf -> 1
        pos[wl_inf & p_inf] = 1.
        return pos

    def responsibilities(self, x):
        r = np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unstable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = self.fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = self.fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0 + self.eps_nan, 1 - self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l  # I do not use this one at the end

    def look_lookup(self, x):
        x = np.array((self.lookup_resolution * x).astype(int))
        x[x < 0] = 0
        x[x == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x]

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)
