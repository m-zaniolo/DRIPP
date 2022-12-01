#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:24:17 2021

@author: martazaniolo
"""

import numpy as np

class node_param:
    def __init__(self):
        self.c = []
        self.b = []
        self.w = []


#class ncRBF(object):

def get_output(inp, param, lin_param, N, M, K):
    # get layers charateristics
    # N = self.N # number of nodes in hidden layer
    # M = self.M # number of inputs
    # K = self.K # number of outputs
    
    phi = []
    o = []
    output = []
    
    for j in range(N):
        bf = 0
        
        for i in range(M):
            num = (inp[i] - param[j].c[i])*(inp[i] - param[j].c[i])
            den = (param[j].b[i]*param[j].b[i])
            
            if den < pow(10,-6):
                den = pow(10,-6)
            
            bf = bf + num / den
        
        phi.append( np.exp(-bf) )
            
    for k in range(K):
        o = lin_param[k]
        for j in range(N):
            o = o + param[j].w[k]*phi[j]
            
        if o > 1:
            o = 1.0
        if o < 0:
            o = 0.0
        
        output.append(o)
        
    return output
        



def set_param(inst_capacity, N, M, K, policies):
    
    param_string = policies[0]
    count = 0
    lin_param = []
    param = []
    planning_param = []
    
    # lin parameters. As many as the outputs
    for k in range(K):
        lin_param.append(param_string[count])
        count += 1
    
    
    # RBF paramters
    for i in range(N): # nodes
        node = node_param()
        for j in range(M):
            node.c.append(param_string[count]) # center
            count += 1
            node.b.append(param_string[count]) # radius
            count += 1
        
        for k in range(K):
            node.w.append(param_string[count]) # output weight
            count += 1
    
        param.append(node)  

        
    return param,lin_param
    
