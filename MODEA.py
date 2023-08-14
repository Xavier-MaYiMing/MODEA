#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/8 16:57
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : MODEA.py
# @Statement : Multi-objective differential evolution algorithm (MODEA)
# @Reference : Ali M, Siarry P, Pant M. An efficient differential evolution based algorithm for solving multi-objective optimization problems[J]. European Journal of Operational Research, 2012, 217(2): 404-416.
import numpy as np
import matplotlib.pyplot as plt


def cal_obj(x):
    # ZDT3
    if np.any(x < 0) or np.any(x > 1):
        return [np.inf, np.inf]
    f1 = x[0]
    num1 = 0
    for i in range(1, len(x)):
        num1 += x[i]
    g = 1 + 9 * num1 / (len(x) - 1)
    f2 = g * (1 - np.sqrt(x[0] / g) - x[0] / g * np.sin(10 * np.pi * x[0]))
    return [f1, f2]


def nd_sort(objs):
    # fast non-domination sort
    (npop, nobj) = objs.shape
    n = np.zeros(npop, dtype=int)  # the number of individuals that dominate this individual
    s = []  # the index of individuals that dominated by this individual
    rank = np.zeros(npop, dtype=int)
    ind = 1
    pfs = {ind: []}  # Pareto fronts
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind
    while pfs[ind]:
        pfs[ind + 1] = []
        for i in pfs[ind]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1
        ind += 1
    pfs.pop(ind)
    return pfs, rank


def crowding_distance(objs, pfs):
    # crowding distance
    (npop, nobj) = objs.shape
    cd = np.zeros(npop)
    for key in pfs.keys():
        pf = pfs[key]
        temp_obj = objs[pf]
        fmin = np.min(temp_obj, axis=0)
        fmax = np.max(temp_obj, axis=0)
        df = fmax - fmin
        for i in range(nobj):
            if df[i] != 0:
                rank = np.argsort(temp_obj[:, i])
                cd[pf[rank[0]]] = np.inf
                cd[pf[rank[-1]]] = np.inf
                for j in range(1, len(pf) - 1):
                    cd[pf[rank[j]]] += (objs[pf[rank[j + 1]], i] - objs[pf[rank[j]], i]) / df[i]
    return cd


def nd_cd_sort(pop, objs, npop):
    # sort the population according to the Pareto rank and crowding distance
    pfs, rank = nd_sort(objs)
    cd = crowding_distance(objs, pfs)
    temp_list = []
    for i in range(len(pop)):
        temp_list.append([pop[i], objs[i], rank[i], cd[i]])
    temp_list.sort(key=lambda x: (x[2], -x[3]))
    next_pop = np.zeros((npop, pop.shape[1]))
    next_objs = np.zeros((npop, objs.shape[1]))
    for i in range(npop):
        next_pop[i] = temp_list[i][0]
        next_objs[i] = temp_list[i][1]
    return next_pop, next_objs


def dominates(obj1, obj2):
    # determine whether obj1 dominates obj2
    sum_less = 0
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:
            return False
        elif obj1[i] != obj2[i]:
            sum_less += 1
    return sum_less > 0


def main(npop, iter, lb, ub, F=0.5, CR=0.3):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param F: scalar number (default = 0.5)
    :param CR: crossover rate (default = 0.3)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = np.array([cal_obj(x) for x in pop])  # objectives
    o_pop = -pop + lb + ub  # opposite population
    o_objs = np.array([cal_obj(x) for x in o_pop])  # the objectives of opposite population
    pop, objs = nd_cd_sort(np.concatenate((pop, o_pop), axis=0), np.concatenate((objs, o_objs), axis=0), npop)

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 20 == 0:
            print('Iteration ' + str(t + 1) + ' completed.')
        new_pop = np.zeros((npop, nvar))

        for i in range(npop):

            # Step 2.1. Select three individuals
            [r1, r2, r3] = np.sort(np.random.choice(npop, 3, replace=False))

            # Step 2.2. Differential evolution
            new_pop[i] = pop[r1] + F * (pop[r2] - pop[r3])
            flag = np.random.random(nvar) > CR
            new_pop[i, flag] = pop[i, flag]

        # Step 2.3. Environmental solution
        new_pop = np.where(((new_pop >= lb) & (new_pop <= ub)), new_pop, pop)
        new_objs = np.array([cal_obj(x) for x in new_pop])
        pop, objs = nd_cd_sort(np.concatenate((pop, new_pop), axis=0), np.concatenate((objs, new_objs), axis=0), npop)

    # Step 3. Sort the results
    dom = np.full(npop, False)
    for i in range(npop - 1):
        for j in range(i, npop):
            if not dom[i] and dominates(objs[j], objs[i]):
                dom[i] = True
            if not dom[j] and dominates(objs[i], objs[j]):
                dom[j] = True
    pf = objs[~dom]
    plt.figure()
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    plt.scatter(x, y)
    plt.xlabel('objective 1')
    plt.ylabel('objective 2')
    plt.title('The Pareto front of ZDT3')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    main(200, 500, np.array([0] * 30), np.array([1] * 30))
