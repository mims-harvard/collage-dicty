import logging
import pickle
import gzip
from os.path import join as pjoin

import numpy as np

import dfmf


logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('Collage - fuse')


class Fuser(object):
    """Collective matrix factorization.

    Parameters
    ----------
    data_path : string
        A path to directory with data matrices.

    random_state : int, RandomState instance, or None (default=0)
        The seed of the pseudo random number generator that is used
        for initialization of latent matrices.

    Attributes
    ----------
    n_run : int
        Number of independent runs of the collective matrix
        factorization.

    R_ : dict in the format of block indices
        A collection of data matrices represented as block entries in
        higher order matrix structure (block indices).

        Collection of data matrices is represented in
        the following format::

            R = {'shape': (r, r),
                (0, 1): R_12, ..., (0, r - 1): R_1r,
                                    ...,
                (r - 1, 0): R_r1, (r - 1, 1): R_r2, ..., (r - 1, r - 2): R_rr-1},

        where r denotes the number of object types modeled by the system.

        Note that dictionary entries should be provided only for pairs
        of object types for which data matrices are available.

    Theta_ : dict in the format of block indices
        A collection of constraint matrices represented as block entries
        in higher order matrix structure (block indices).

        The format of a collection of constraint matrices
        is the following::

            Theta = {'shape': (r, r),
                    (0, 0): [Theta_1^(1), Theta_1^(2), ...],
                                    ...,
                    (r - 1, r - 1): [Theta_r^(1), Theta_r^(2), ...]},

        where r is the number of object types in the system.

        Note that dictionary entries are only over object types for which
        constraint matrices are available.

    ns_ : {list-like}
        A vector of the number of objects of every object type.

    cs_ : {list-like}
        A vector of factorization ranks, one value for each object type.

    G_ : dict in the format of block indices
        Recipe matrices estimated by matrix factorization-based data fusion.

        Recipe matrices are represented in the format::

            G = {'shape': (r, r),
                (0,0): G_1, (1,1): G_2, ... (r - 1, r - 1): G_r},

        where r is the number of object types in the system. One recipe
        matrix is provided for each object type in the system. In the
        case of multiple runs of factorization algorithm G_ contains
        recipe matrices estimated in the last run.

    S_ : dict in the format of block indices
        Backbone matrices estimation by matrix factorization-based
        data fusion.

        Backbone matrices are represented in the format::

            S = {'shape': (r, r),
                (0, 1): S_12, ..., (0, r - 1): S_1r,
                                    ...,
                (r - 1, 0): S_r1, (r - 1, 1): S_r2, ..., (r - 1, r - 2): S_rr-1},

        where r is the number of object types in the system.

        Note that dictionary entries are provided for all pairs of object
        types which have the corresponding data matrices. In the case of
        multiple runs of factorization algorithm S_ contains backbone
        matrices estimated in the last run.
    """
    def __init__(self, data_path, random_state=0):
        self.data_path = data_path
        self.load_data()
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)
        self.R_ = None
        self.Theta_ = None
        self.ns_ = None
        self.cs_ = None
        self.G_ = None
        self.S_ = None

    def load_data(self):
        """Load a collection of data matrices that are considered
        for bacterial response gene prioritization.
        """
        _log.info('Loading data matrices')
        self.R_14, self.g_map = self._load_matrix(
            'R_14.tsv.gz', row_map=True)
        self.R_12 = self._load_matrix('R_12.tsv.gz')
        self.R_24 = self._load_matrix('R_24.tsv.gz')
        self.R_23 = self._load_matrix('R_23.tsv.gz')
        self.R_54 = self._load_matrix('R_54.tsv.gz')
        self.R_64 = self._load_matrix('R_64.tsv.gz')
        self.R_65 = self._load_matrix('R_65.tsv.gz')
        self.R_15 = self._load_matrix('R_15.tsv.gz')
        self.R_16 = self._load_matrix('R_16.tsv.gz')
        self.R_17 = self._load_matrix('R_17.tsv.gz')
        self.R_18 = self._load_matrix('R_18.tsv.gz')
        self.R_19 = self._load_matrix('R_19.tsv.gz')
        self.R_110 = self._load_matrix('R_110.tsv.gz')
        self.T_1 = self._load_matrix('Theta1.tsv.gz')

    def _load_matrix(self, fname, row_map=False):
        """Load one data matrix.

        Parameters
        ----------
        fname : string
            The name of the file with data matrix.

        row_map : boolean
            If true then also identifiers of row objects are
            read from file.
        """
        X = np.genfromtxt(pjoin(
            self.data_path, fname), delimiter='\t', skip_header=1,
            missing_values='', filling_values='0')
        if row_map:
            row_names = np.genfromtxt(pjoin(
                self.data_path, fname), delimiter='\t', skip_header=1,
                usecols=[0], dtype=['S12'])
            r_map = {e[0]: i for i, e in enumerate(row_names)}
            return X[:, 1:], r_map
        return X[:, 1:]

    def fuse(self, n_run=20, dump=True):
        """Interface to collective matrix factorization algorithm.

        Parameters
        ----------
        n_run : int
            Number of independent factorization runs to
            be performed (default value is 20).
        dump : boolean
            If true (default) then estimated latent matrices
            is saved to disk for later use.
        """
        self.n_run = n_run
        self.Theta_ = {'shape': (10, 10), (0, 0): [self.T_1]}
        self.R_ = {
            'shape': (10, 10), (0, 3): self.R_14, (0, 1): self.R_12,
            (1, 3): self.R_24, (1, 2): self.R_23, (4, 3): self.R_54,
            (5, 3): self.R_64, (5, 4): self.R_65, (0, 4): self.R_15,
            (0, 5): self.R_16, (0, 6): self.R_17, (0, 7): self.R_18,
            (0, 8): self.R_19, (0, 9): self.R_110
        }
        self.ns_ = (
            self.R_12.shape[0], self.R_12.shape[1], self.R_23.shape[1],
            self.R_14.shape[1], self.R_54.shape[0], self.R_16.shape[1],
            self.R_17.shape[1], self.R_18.shape[1], self.R_19.shape[1],
            self.R_110.shape[1]
        )
        proc = 0.1
        self.cs_ = map(lambda x: max(5, int(proc * x)), self.ns_)
        for r in xrange(self.n_run):
            _log.info('[%d] Run' % r)
            this_G, this_S, err_system = dfmf.dfmf(
                self.R_, self.Theta_, self.ns_, self.cs_,
                max_iter=200, init_typ='random',
                system_eps=1e-2, compute_err=True,
                return_err=True, random_state=self.random_state)
            self.G_ = this_G
            self.S_ = this_S
            if dump:
                dump_gene_map = r == 0
                self.dump('%d-' % r, dump_gene_map)

    def dump(self, prefix=None, dump_gene_map=True):
        """Use Python cPickle to dump estimated latent matrices
        to files for later use.

        Parameters
        ----------
        prefix : string or None (default)
            The prefix of file names.

        dump_gene_map : boolean
            If true (default) then a map between gene names
            and rows is saved to disk
        """
        _log.info('Dumping latent matrices')
        pickle.dump(self.G_, gzip.open(pjoin(
            self.data_path, '%sG.pkl.gz' % prefix), 'w'))
        pickle.dump(self.S_, gzip.open(pjoin(
            self.data_path, '%sS.pkl.gz' % prefix), 'w'))
        if dump_gene_map:
            pickle.dump(self.g_map, gzip.open(pjoin(
                self.data_path, 'g_map.pkl.gz'), 'w'))
