#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lake class contains methods to simulate water reservoir in the Santa Barbara watershed. 
"""

import numpy as np



class Lake(object):
        
    def storage_to_level(self, s):
        return np.interp(s, self.lsv[2,:], self.lsv[0,:])
    
    def storage_to_area(self, s):
        return np.interp(s, self.lsv[2,:], self.lsv[1,:])
    
    def level_to_storage(self, l):
        return np.interp(l, self.lsv[0,:], self.lsv[2,:])
    
    def actual_release(self, s, u):
        mr = self.min_rel(s)
        Mr = self.max_rel(s)
        return min(Mr, u), mr  
    
    def integration(self, s0, u, n0, demand, e=0):
        HH     = self.integration_step
        rr     = [-999]
        ss     = [s0]
        A = self.storage_to_area(s0)

            
        for h in range(HH):
            r_, mr = self.actual_release(ss[h], u*demand/HH)
            rr.append(r_)
            spill = max(0, mr-r_)
            s_ = ss[h] + self.deltaH*( n0/HH - rr[h+1] - spill ) - e*A/1000/HH
            s_ = min(s_, self.smax) 
            ss.append(s_)

        s = ss[-1] 
        r = np.sum(rr[1:]) 
        return s, r
    
    def sim_lake(self, s0, r, e=0):
        s = [s0]
        infl = self.inflow
        u = (r/self.max_release)*2-1
        r_c = [-999]
        for t in range(self.H):
            s_, r_  = self.integration(s[t], u[t], infl[t], e)
            s.append(s_)
            r_c.append(r_)
        
        return s, r_c
            















