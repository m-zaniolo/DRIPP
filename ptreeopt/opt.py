from __future__ import division

import copy
import datetime
import functools
import logging
import time
import pickle
import random
import numpy as np

from .tree import PTree
from ptreeopt.executors import SequentialExecutor

logger = logging.getLogger(__name__)


def function_runner(func, solution):
    # model.f has side effects: it changes values on P
    # so for parallel running  we want to return
    # also the modified P
    logger.debug("trying to run {} for {}".format(func, solution))
    results = func(solution)
    logger.debug("succesfully ran {} with {}: {}".format(func, solution,
                                                         results))

    return solution, results

class PTreeOpt(object):
    '''Algorithm for optimizing policy trees

    Parameters
    ----------
    f : callable
    feature_bounds :
    discrete_actions : boolean, optional
    action_bounds :
    action_names :
    population size : int, optional
    mu : float, optional
    max_depth : int, optional
    mut_prob : float, optional
    cx_prob : float, optional
    feature_names :
    discrete_features :
    multiobj : bool, optional
    epsilons :


    Raises
    ------
    ValueError

    '''

    process_log_message = ('{} nfe; {} sec; '
                '{} M$/year')

    def __init__(self, f, feature_bounds, discrete_actions=False,
                 action_bounds=None, action_names=None,
                 population_size=100, mu=15, max_depth=4, mut_prob=0.5,
                 cx_prob=0.9, feature_names=None,
                 discrete_features=None, multiobj=True, epsilons=None, num_policies = 1):

        self.f = functools.partial(function_runner, f)
        self.num_features = len(feature_bounds)
        self.feature_bounds = feature_bounds
        self.discrete_actions = discrete_actions
        self.action_bounds = action_bounds
        self.action_names = action_names
        self.popsize = population_size
        self.mu = mu
        self.max_depth = max_depth
        self.mut_prob = mut_prob
        self.cx_prob = cx_prob
        self.feature_names = feature_names
        self.discrete_features = discrete_features
        self.multiobj = multiobj
        self.epsilons = epsilons
        self.num_policies = 5

        if feature_names is not None and\
           len(feature_names) != len(feature_bounds):
            raise ValueError(('feature_names and feature_bounds '
                              'must be the same length.'))

        if discrete_features is not None and\
           len(discrete_features) != len(feature_bounds):
            raise ValueError(('discrete_features and feature_bounds '
                              'must be the same length.'))

        if discrete_actions:
            if action_names is None or action_bounds is not None:
                raise ValueError(('discrete_actions must be run with '
                                  'action_names, (which are strings), '
                                  'and not action_bounds.'))
        else:
            if action_bounds is None:
                raise ValueError(('Real-valued actions (which is the case by '
                                'default, discrete_actions=False) must include' 
                                'action_bounds. Currently only one action is '
                                'supported, so bounds = [lower, upper].'))

        if mu > population_size:
            raise ValueError(('Number of parents (mu) cannot be greater '
                              'than the population_size.'))

    def iterate(self):
        # TODO:: have to separate selection
        # one selection function for multi objective
        # one selection function for single objective

        # selection: find index numbers for parents
        if not self.multiobj:

            parents = self.select_truncation(self.objectives)

            if self.best_f is None or self.objectives[parents[0]] < self.best_f:
                self.best_f = self.objectives[parents[0]]
                self.best_p = copy.deepcopy(self.population[parents[0]])

        else:
            parents = [self.binary_tournament(self.population, self.objectives)
                       for _ in range(10)]
            if self.best_f is None:

                self.best_f = self.objectives[parents]
                self.best_p = self.population[parents,:]

            self.best_p, self.best_f = self.archive_sort(self.best_p,
                                                 self.best_f, self.population,
                                                 self.objectives)


        # New strategy MZ: add best_p to new_pop and make them parents, all the
        # rest in children


        i = 0
        self.population[i] = copy.deepcopy(self.best_p)
        i += 1

        while i < self.popsize:
            child = [[],[],[],[],[]]
            if np.random.rand() < 0.9: #self.cx_prob: #either mutate or crossover
                if np.random.rand() < 0.5: #crossover
                    P1, P2 = self.population[ np.random.choice(parents, 2) ]
                    for tree in range(5):
                        child[tree] = self.crossover(P1[tree], P2[tree])[0]
                    # bloat control
                        while child[tree].get_depth() > self.max_depth:
                            child[tree] = self.crossover(P1[tree], P2[tree])[0]
                        child[tree].prune()
                else: #mutation
                    PP = self.population[np.random.choice(parents, 1)]
                    for tree in range(5):
                        #np.random.randint(len(self.best_p))#choice(self.best_p, 1)
                        child[tree] = self.mutate( PP[0][tree], tree ) 
                        child[tree].prune()
            else:  # replace with randomly generated child
                child = self.random_individual()

            child = self.check_actions(child)
            self.population[i] = copy.deepcopy(child)
            i += 1

    def run(self, max_nfe=1000, log_frequency=100, snapshot_frequency=100,
            executor=SequentialExecutor(), extend_opt = [], drought_type = [], seed = 0, action_file = 'baseline'):
        '''Run the optimization algorithm

        Parameters
        ----------
        max_nfe : int, optional
        log_frequency :  int, optional
        snapshot_frequency : int or None, optional
                             int specifies frequency of storing convergence
                             information. If None, no convergence information
                             is retained.
        executor : subclass of BaseExecutor, optional

        Returns
        -------
        best_p
            best solution or archive in case of many objective
        best_f
            best score(s)
        snapshots
            if snapshot_frequency is not None, convergence information

        '''

        start_time = time.time()
        nfe, last_log, last_snapshot = 0, 0, 0

        self.best_f = None
        self.best_p = None

        self.population = np.array( [self.random_individual() for _ in range(self.popsize)])
        if len(extend_opt):
            for p in range(len(extend_opt)):
                self.population[p] = extend_opt[p]


        if snapshot_frequency is not None:
            snapshots = {'nfe': [], 'time': [], 'best_f': [],
                         'best_P': [], 'objectives': [], 'pop': []}
        else:
            snapshots = None

        while nfe < max_nfe:
            for member in self.population[0]:
                member.clear_count() # reset action counts to zero

            # evaluate objectives
            population, objectives = executor.map(self.f, self.population)
            #print(objectives)
            self.objectives = objectives
            self.population = np.asarray(population)

            
            for member in population:
                member[0].clear_count() # reset action counts to zero
                member[1].clear_count()
                member[2].clear_count()
                member[3].clear_count()
                member[4].clear_count()
                #member.normalize_count() # convert action count to percent

            nfe += self.popsize

            self.iterate()

            if nfe >= last_log + log_frequency:
                last_log = nfe
                elapsed = datetime.timedelta(
                    seconds=time.time() - start_time).seconds

                if not self.multiobj:
                    logger.info(self.process_log_message.format(nfe,
                                    elapsed, self.best_f))
                else:
                    # TODO:: to be tested
                    logger.info('# nfe = %d\n%s\n%s' % (nfe, self.best_f,
                                                    self.best_f.shape))

            if nfe >= last_snapshot + snapshot_frequency:
                last_snapshot = nfe
                snapshots['nfe'].append(nfe)
                snapshots['time'].append(elapsed)
                snapshots['best_f'].append(self.best_f)
                snapshots['best_P'] = self.best_p
                snapshots['model'] = self.f
                snapshots['objectives'].append(self.objectives)
                snapshots['pop'] = self.population

                string = 'results/intermediate_results/snapshot_p' + str(drought_type[0]) + '_i' + str(drought_type[1])+'_n'+str(drought_type[2])+ action_file + '_s' + str(seed)

                with open(string, 'wb') as f:
                    pickle.dump(snapshots, f)



        if snapshot_frequency:
            return self.best_p, self.best_f, snapshots
        else:
            return self.best_p, self.best_f

    def random_tree(self, terminal_ratio=0.5, depth = None, action_type = -999):
        '''

        Parameters
        ----------
        terminal_ratio : float, optional

        '''

        if depth:
            a = 0
        else:
            depth = np.random.randint(1, self.max_depth + 1)
        L = []
        S = [0]
        
        flag = 0 #MZ: at least one action is nothing in the initialization

        while S:
            current_depth = S.pop()

            # action node
            if current_depth == depth or (current_depth > 0 and\
                                      np.random.rand() < terminal_ratio):
                if self.discrete_actions:
                    if action_type > -999:
                        if flag == 0: 
                            L.append([str(np.random.choice(self.action_names[action_type]))])
                            flag += 1
                        else:
                            L.append(['nothing'])
                        
                    else:
                        L.append([str(np.random.choice(self.action_names))])
                else:
                    L.append([np.random.uniform(*self.action_bounds)])

            else:
                x = np.random.choice(self.num_features)
                v = np.random.uniform(*self.feature_bounds[x])
                L.append([x, v])
                S += [current_depth + 1] * 2

        T = PTree(L, self.feature_names, self.discrete_features)
        T.prune()
        return T

    def random_rm_tree(self, ref_tree, terminal_ratio=0.5):
        '''

        Parameters
        ----------
        terminal_ratio : float, optional

        '''

        depth = ref_tree.get_depth()
        L = []
        S = [0]
        action_pool = []
        for node in ref_tree.L:
            if node.is_feature == False:
                action_pool.append(node.value)
        
        i = len(action_pool)-1
        while S:
            current_depth = S.pop()

            # action node
            if current_depth == depth or (current_depth > 0 and\
                                      np.random.rand() < terminal_ratio):
                
                L.append([str(action_pool[i])]) #same actions but in reverse so that high storage isn't paired with removal
                i -= 1

            else:
                x = np.random.choice(self.num_features)
                v = np.random.uniform(*self.feature_bounds[x])
                L.append([x, v])
                S += [current_depth + 1] * 2

        T = PTree(L, self.feature_names, self.discrete_features)
        T.prune()
        return T

    def check_actions(self, individual):
        # make sure remove policy has no additional actions as corresponding build tree

        action_b0 = []
        for node in individual[0].L:
            if node.is_feature == False:
                action_b0.append(node.value)
        b0 = set(np.unique(action_b0))

        action_r0 = []        
        for node in individual[3].L:
            if node.is_feature == False:
                action_r0.append(node.value)
        r0 = set(np.unique(action_r0))
        
        if b0 != r0 : 
            individual[3] = self.random_rm_tree(individual[0])
            
        action_b1 = []
        for node in individual[1].L:
            if node.is_feature == False:
                action_b1.append(node.value)
        b1 = set(np.unique(action_b1))

        action_r1 = []        
        for node in individual[4].L:
            if node.is_feature == False:
                action_r1.append(node.value)
        r1 = set(np.unique(action_r1))
        
        if b1 != r1 : 
            individual[4] = self.random_rm_tree(individual[1])
            
        return individual
                
        
                


    def random_individual(self):
        ensemble = [self.random_tree(action_type = i) for i in range(3)]
        ensemble.extend( [self.random_rm_tree(ensemble[i]) for i in range(2) ] )
        return ensemble

    def select_truncation(self, obj):
        return np.argsort(obj)[:self.mu]

    def crossover(self, PP1, PP2):
        P1, P2 = [copy.deepcopy(P) for P in (PP1, PP2)]
        # should use indices of ONLY feature nodes
        feature_ix1 = [i for i in range(P1.N) if P1.L[i].is_feature]
        feature_ix2 = [i for i in range(P2.N) if P2.L[i].is_feature]
        index1 = np.random.choice(feature_ix1)
        index2 = np.random.choice(feature_ix2)
        slice1 = P1.get_subtree(index1)
        slice2 = P2.get_subtree(index2)
        P1.L[slice1], P2.L[slice2] = P2.L[slice2], P1.L[slice1]
        P1.build()
        P2.build()
        return (P1, P2)

    def mutate(self, P, tree, mutate_actions=True):
        PP = copy.deepcopy(P)

        for item in PP.L:
            if np.random.rand() < self.mut_prob:
                if item.is_feature:
                    if np.random.rand() < 0.3: #mutate feature nature
                        item.index = np.random.choice(self.num_features)
                        item.threshold = np.random.uniform(*self.feature_bounds[item.index])
                        item.name = self.feature_names[item.index]
                    else:
                        low, high = self.feature_bounds[item.index]
                        if item.is_discrete:
                            item.threshold = np.random.randint(low, high+1)
                        else:
                            item.threshold = self.bounded_gaussian(
                                item.threshold, [low, high])
                elif all([mutate_actions, tree<3]):
                    if self.discrete_actions:
                        item.value = str(np.random.choice(self.action_names[tree]))
                    else:
                        item.value = self.bounded_gaussian(
                            item.value, self.action_bounds)

        return PP

    def mutate_structure(self, P):
        PP = copy.deepcopy(P)
        feature_ix1 = [i for i in range(PP.N) if PP.L[i].is_feature]
        index1 = np.random.choice(feature_ix1)
        slice1 = PP.get_subtree(index1)
        dd = slice1.get_depth()
        slice2 = self.random_tree(depth = dd)
        PP.L[slice1] = PP.L[slice2]
        return PP


    def bounded_gaussian(self, x, bounds):
        # do mutation in normalized [0,1] to avoid sigma scaling issues
        lb, ub = bounds
        xnorm = (x - lb) / (ub - lb)
        x_trial = np.clip(xnorm + np.random.normal(0, scale=0.05), 0, 1)

        return lb + x_trial * (ub - lb)

    def dominates(self, a, b):
        a = a // self.epsilons
        b = b // self.epsilons
        # assumes minimization
        # a dominates b if it is <= in all objectives and < in at least one
        return (np.all(a <= b) and np.any(a < b))

    def same_box(self, a, b):
        if self.epsilons:
            a = a // self.epsilons
            b = b // self.epsilons
        return np.all(a == b)

    def binary_tournament(self, P, f):
        # select 1 parent from population P
        # (Luke Algorithm 99 p.138)
        i = np.random.randint(0, P.shape[0], 2)
        a, b = f[i[0]], f[i[1]]
        if self.dominates(a, b):
            return i[0]
        elif self.dominates(b, a):
            return i[1]
        else:
            return i[0] if np.random.rand() < 0.5 else i[1]

    def archive_sort(self, A, fA, P, fP):
        A = np.vstack((A, P))
        fA = np.vstack((fA, fP))
        N = len(A)
        keep = np.ones(N, dtype=bool)

        for i in range(N):
            for j in range(i + 1, N):
                if keep[j] and self.dominates(fA[i, :], fA[j, :]):
                    keep[j] = False

                elif keep[i] and self.dominates(fA[j, :], fA[i, :]):
                    keep[i] = False

                elif self.same_box(fA[i, :], fA[j, :]):
                    keep[j] = False #np.random.choice([i, j])] = False

        return (A[keep], fA[keep, :])
