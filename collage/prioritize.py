import logging
import pickle
import gzip
import os
from operator import itemgetter
from os.path import join as pjoin

import numpy as np
from scipy import stats


logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('Collage - prioritize')


class Prioritizer(object):
    """Gene prioritization by collective matrix factorization.

    Parameters
    ----------
    path : string
        A path to directory where latent matrices were saved
        by the Fuser.

    seed_fname : string
        Name of the file in data directory that contains seed genes.

    res_fname : string
        Name of the file where gene prioritization results are saved.

    random_state : int, RandomState instance, (0 by default)
        The seed of the pseudo random number generator that is used
        for initialization of latent matrices.

    Attributes
    ----------
    gene2p_ : dict
        Scored candidate genes.

    gene2pv_ : dict
        P-values of candidate genes.
    """
    def __init__(self, path, seed_fname, res_fname, random_state=0):
        self.path = path
        self.seed_fname = seed_fname
        self.res_fname = res_fname
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)
        self._read_seeds()
        self._load_factors()
        self.gene2p_, self.gene2pv_ = {}, {}

    def _read_seeds(self):
        """Read seed genes that are used for prioritization."""
        fname = pjoin(self.path, self.seed_fname)
        self.seeds = [line.strip().split('\t')[0] for line in open(fname)]
        _log.info('Seed genes: %s' % ', '.join(self.seeds))

    def _load_factors(self):
        """Read latent matrices obtained by collective matrix factorization."""
        _log.info('Loading latent matrices')
        self.G_, self.S_ = [], []
        for fname in os.listdir(self.path):
            fpath = pjoin(self.path, fname)
            if os.path.isfile(fpath) and fname.endswith('G.pkl.gz'):
                _log.info('Loading recipe matrices: %s' % fname)
                self.G_.append(pickle.load(gzip.open(fpath)))

                fname = fname.replace('G', 'S')
                _log.info('Loading backbone matrices: %s' % fname)
                spath = pjoin(self.path, fname)
                self.S_.append(pickle.load(gzip.open(spath)))

            if os.path.isfile(fpath) and fname == 'g_map.pkl.gz':
                _log.info('Loading gene map')
                self.g_map = pickle.load(gzip.open(fpath))
                self.inv_g_map = {i: g for g, i in self.g_map.iteritems()}
        self.n_run = len(self.G_)
        _log.info('Loaded latent matrices from %d runs' % self.n_run)

    def _load_ddbg2name(self):
        """Read a map between Dictyostelium DDBGs and gene names."""
        _log.info('Reading DDB-GeneID-UniProt.txt')
        f = open(pjoin(self.path, "DDB-GeneID-UniProt.txt"))
        f.readline()
        ddbg2name = dict(line.strip('\t').split()[1:3] for line in f)
        f.close()
        return ddbg2name

    def chain_factors(self):
        """Gene profiling by chaining of latent matrices."""
        self.chained_factors = []
        for r in xrange(self.n_run):
            G = self.G_[r]
            S = self.S_[r]
            # Following are chains of latent matrices as defined by the data fusion
            # graph used in bacterial response gene prioritization for Dictyostelium.
            f = [G[0, 0], np.dot(G[0, 0], S[0, 6]), np.dot(G[0, 0], S[0, 9]),
                 np.dot(G[0, 0], S[0, 7]), np.dot(G[0, 0], S[0, 8]),
                 np.dot(G[0, 0], S[0, 1]), np.dot(G[0, 0], S[0, 5]),
                 np.dot(G[0, 0], S[0, 4]), np.dot(G[0, 0], S[0, 3]),
                 np.dot(G[0, 0], np.dot(S[0, 1], S[1, 2])),
                 np.dot(G[0, 0], np.dot(S[0, 5], S[5, 4])),
                 np.dot(G[0, 0], np.dot(S[0, 5], S[5, 3])),
                 np.dot(G[0, 0], np.dot(S[0, 1], S[1, 3])),
                 np.dot(G[0, 0], np.dot(S[0, 4], S[4, 3])),
                 np.dot(G[0, 0], np.dot(S[0, 5], np.dot(S[5, 4], S[4, 3])))]
            self.chained_factors.append(f)

    def prioritize(self, n_permute=50):
        """Interface to similarity estimation, similarity scoring,
        score aggregation and gene ranking.

        Parameters
        ----------
        n_permute : int
            The number of times that seed set is randomly drawn from
            the pool of all candidate genes. Random seed sets are
            used to estimate the empirical P-values of scored
            candidate genes.
        """
        self.n_permute = n_permute
        self.chain_factors()
        self._y = np.zeros(len(self.g_map))
        _log.info('Total genes: %d' % len(self.g_map))

        for gene in self.seeds:
            self._y[self.g_map[gene]] = 1
        _log.info('Seed genes: %d' % np.sum(self._y == 1))

        train = np.nonzero(self._y == 1)[0]
        loo = [(np.delete(train, i), np.array([idx])) for
               i, idx in enumerate(train)]

        _log.info('Scoring seed genes')
        self._prioritize(loo)

        _log.info('Scoring candidate genes')
        test = np.nonzero(self._y == 0)[0]
        self._prioritize([(train, test)])

        _log.info('Writing results to file')
        self.pp_list()

    def _prioritize(self, train_test):
        """Helper method for gene scoring.

        Parameters
        ----------
        train_test : {list-like} of tuples
            Training (seed) and test (candidate) genes.
        """
        for train_idx, test_idx in train_test:
            self.predict(train_idx, test_idx, True)
        train_size = len(train_test[0][0])
        test_idx = np.hstack([te for _, te in train_test])
        self.pvalue(train_size, test_idx)

    def pvalue(self, train_size, test_idx):
        """Estimation of empirical P-values.

        Parameters
        ----------
        train_size : int
            The number of seed genes

        test_idx : numpy.ndarray
            A vector of row indices in profile matrices that
            correspond to candidate genes.
        """
        _log.info('P-value estimation')
        perm_preds = []
        for i in xrange(self.n_permute):
            _log.info('[%d/%d] P-value estimation' % (i + 1, self.n_permute))
            perm_preds.append(self.pvalue_worker(train_size, test_idx))
        pred = np.array([self.gene2p_[self.inv_g_map[t_idx]]
                         for t_idx in test_idx])
        counts = [(perm_pred - pred) > 0 for perm_pred in perm_preds]
        p_vals = np.sum(np.array(counts), 0) / float(self.n_permute)
        assert p_vals.shape == test_idx.shape, 'Dimension mismatch'
        for pv, n_idx in zip(p_vals, test_idx):
            self.gene2pv_[self.inv_g_map[n_idx]] = pv

    def pvalue_worker(self, train_size, test_idx):
        """Selection of random seed set from the pool of candidate
        genes and re-scoring genes based on this random set of seed genes.

        Parameters
        ----------
        train_size : int
            The number of seed genes

        test_idx : numpy.ndarray
            A vector of row indices in profile matrices that
            correspond to candidate genes.
        """
        perm = np.nonzero(self._y == 0)[0]
        perm_train_idx = self.random_state.permutation(perm)[:train_size]
        return self.predict(perm_train_idx, test_idx, False)

    def predict(self, train_idx, test_idx, save):
        """Similarity estimation, similarity scoring and score aggregation.

        Parameters
        ----------
        train_idx : numpy.ndarray
            A vector of row indices in profile matrices that
            correspond to seed genes.

        test_idx : numpy.ndarray
            A vector of row indices in profile matrices that
            correspond to candidate genes.

        save : boolean
            An indicator whether to save candidate scores or not.
        """
        _log.info('Prediction')
        pred_final = np.zeros((len(test_idx), self.n_run))
        for r in xrange(self.n_run):
            _log.info('[%d/%d] Run' % (r + 1, self.n_run))
            chained_factors = self.chained_factors[r]
            for i, te in enumerate(test_idx):
                if i % 1000 == 0:
                    _log.info('[%d/%d] Prediction' % (i, len(test_idx)))
                sm = [[stats.spearmanr(chain[te], chain[tr])[0]
                       for chain in chained_factors] for tr in train_idx]
                sm = [np.median(c) for c in sm]
                pred_final[i, r] = np.median(sm)
        pred_final = np.mean(pred_final, 1)
        assert pred_final.shape == test_idx.shape, 'Dimension mismatch'
        if save:
            for pf, t_idx in zip(pred_final, test_idx):
                self.gene2p_[self.inv_g_map[t_idx]] = pf
        return pred_final

    def pp_list(self):
        """Save results of gene prioritization to a file. """
        ddbg2name = self._load_ddbg2name()
        fname = pjoin(self.path, self.res_fname)
        f = open(fname, 'w')
        f.write('Gene\tDictybase\tScore\tP-value\n')
        gp = sorted(self.gene2p_.items(), reverse=True, key=itemgetter(1))
        for gene, score in gp:
            pv = self.gene2pv_[gene]
            dicty_url = 'http://dictybase.org/gene/%s' % gene
            pp_g = ddbg2name.get(gene, gene)
            f.write('%s\t%s\t%f\t%f\n' % (pp_g, dicty_url, score, pv))
        f.close()
