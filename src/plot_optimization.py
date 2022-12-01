#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:34:40 2021

@author: martazaniolo
"""

import matplotlib
import numpy as np
from hydro_simulation import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.matlib as mat


def plot_pareto(objs, nseeds = 1):
    plt.style.use('seaborn-darkgrid')

    if nseeds == 1:
        plt.scatter([o[0] for o in objs],[o[1] for o in objs])
    else:        
        for s in range(nseeds): #assuming multiple runs    
            plt.scatter([o[0] for o in objs[s]],[o[1] for o in objs[s]])
            
    plt.show()
            
    
    
def plot_tr(traj):
       
    plt.style.use('seaborn-darkgrid')
    
    H = np.size(traj.rc)
    
    fig, axs = plt.subplots(2,2)
#    fig.suptitle('Trajectories')
    axs[0,0].plot(traj.sc)
    axs[0,0].set_title('Cachuma allocation')
    axs[0,0].set_xlabel('Time')
    axs[0,0].set_ylabel('Storage [AF]')
    # axs[0,0].set_ylim([0, 20000])
    
    axs[1,0].plot(traj.rc)
    axs[1,0].set_title('Cachuma release')
    axs[1,0].set_xlabel('Time')
    axs[1,0].set_ylabel('Release [AF/d]')
    # axs[1,0].set_ylim([0, 45])
    
    axs[0,1].plot(traj.sgi)
    axs[0,1].set_title('Gibraltar storage')
    axs[0,1].set_xlabel('Time')
    axs[0,1].set_ylabel('Storage [AF]')
    
    axs[1,1].plot(traj.rgi)
    axs[1,1].set_title('Gibraltar release')
    axs[1,1].set_xlabel('Time')
    axs[1,1].set_ylabel('Release [AF/d]')

    
    plt.tight_layout()
    # plt.savefig('../figures/traj_carry.png',  dpi=100)
    plt.show()

    fig, axs = plt.subplots(2,2)
    axs[0,1].plot(traj.sgw)
    axs[0,1].set_title('Groundwater storage')
    axs[0,1].set_xlabel('Time')
    axs[0,1].set_ylabel('Storage [AF]')
    axs[0,1].tick_params(labelleft='off')
    # axs[0,1].set_ylim([0, 11000])
    
    axs[1,1].plot(traj.rgw)
    axs[1,1].set_title('Groundwater pumping')
    axs[1,1].set_xlabel('Time')
    axs[1,1].tick_params(labelleft='off')
    # axs[1,1].set_ylim([0, 20])
    
    axs[0,0].plot(traj.sswp)
    axs[0,0].set_title('SWP allocation')
    axs[0,0].set_xlabel('Time')
    axs[0,0].set_ylabel('Storage [AF]')
    
    axs[1,0].plot(traj.rswp)
    axs[1,0].set_title('SWP withdrawal')
    axs[1,0].set_xlabel('Time')
    axs[1,0].set_ylabel('Release [AF/d]')

    plt.tight_layout()
    plt.show()

    x=range(1,H+1)
    r_d = mat.repmat(traj.rd,1, H)
    r = [ r_d.tolist()[0], traj.rc, traj.rgi, traj.rt, traj.rgw,  traj.rswp, traj.r_mw ]
    plt.stackplot(x, r, labels=['Desal', 'Cachuma', 'Gibraltar', 'Tunnel', 'Groundwater', 'State Water Project', 'Market Water'])
    plt.legend(loc = 'lower left')
    plt.xlabel('Time')
    plt.ylabel('Demand [AFd]')
    plt.title('Water Demand')
    # plt.savefig('../figures/demand_carry.png',  dpi=100)
    plt.show()
    a = np.sum(np.sum(np.matrix(r)))
    print("demand = ")
    print(a)
                  
    c_d = mat.repmat(traj.cd, 1, H)
    # dis_c = mat.repmat(distr_c, 1, H)
    c = [c_d.tolist()[0], traj.cc, traj.cgi, traj.ct, traj.cgw,  traj.cswp, traj.cmw]
    plt.stackplot(x, c, labels=['Desal',  'Cachuma', 'Gibraltar', 'Tunnel', 'Groundwater', 'State Water Project', 'Market Water'])
    plt.legend(loc = 'upper left')
    plt.title('Cost')
    plt.xlabel('Time')
    plt.ylabel('Cost [$]')
    a = np.sum(np.sum(np.matrix(c)))
    print("cost = ")
    print(a)
    # plt.savefig('../figures/cost_carry.png',  dpi=100)
    plt.show()

    
            
def plot_trajectories(param, opt_par, s):
    sns.set_palette("Set1")

    sim = Simulation(opt_par)
    H = sim.H
    sc, rc, sgi, rgi, sgw, rgw, rd, sswp, rswp, r_mw, rt = sim.get_traj(param)

    
    plt.style.use('seaborn-darkgrid')

    fig, axs = plt.subplots(2,2)
#    fig.suptitle('Trajectories')
    axs[0,0].plot(sc)
    axs[0,0].set_title('Cachuma allocation')
    axs[0,0].set_xlabel('Time')
    axs[0,0].set_ylabel('Storage [AF]')
    # axs[0,0].set_ylim([0, 20000])
    
    axs[1,0].plot(rc)
    axs[1,0].set_title('Cachuma release')
    axs[1,0].set_xlabel('Time')
    axs[1,0].set_ylabel('Release [AF/d]')
    # axs[1,0].set_ylim([0, 45])
    
    axs[0,1].plot(sgi)
    axs[0,1].set_title('Gibraltar storage')
    axs[0,1].set_xlabel('Time')
    axs[0,1].set_ylabel('Storage [AF]')
    
    axs[1,1].plot(rgi)
    axs[1,1].set_title('Gibraltar release')
    axs[1,1].set_xlabel('Time')
    axs[1,1].set_ylabel('Release [AF/d]')

    
    plt.tight_layout()
    # plt.savefig('../figures/traj_carry.png',  dpi=100)
    plt.show()

    fig, axs = plt.subplots(2,2)
    axs[0,1].plot(sgw)
    axs[0,1].set_title('Groundwater storage')
    axs[0,1].set_xlabel('Time')
    axs[0,1].set_ylabel('Storage [AF]')
    axs[0,1].tick_params(labelleft='off')
    # axs[0,1].set_ylim([0, 11000])
    
    axs[1,1].plot(rgw)
    axs[1,1].set_title('Groundwater pumping')
    axs[1,1].set_xlabel('Time')
    axs[1,1].tick_params(labelleft='off')
    # axs[1,1].set_ylim([0, 20])
    
    axs[0,0].plot(sswp)
    axs[0,0].set_title('SWP allocation')
    axs[0,0].set_xlabel('Time')
    axs[0,0].set_ylabel('Storage [AF]')
    
    axs[1,0].plot(rswp)
    axs[1,0].set_title('SWP withdrawal')
    axs[1,0].set_xlabel('Time')
    axs[1,0].set_ylabel('Release [AF/d]')



    plt.tight_layout()
    # plt.savefig('../figures/traj_carry.png',  dpi=100)
    plt.show()

 #   rgi = [min(r,sim.gibraltar.max_city) for r in rgi]
#    rc = [min(r,sim.cachuma.max_city) for r in rc]
    
    x=range(1,H+1)
    r_d = mat.repmat(rd,1, H)
    r = [ r_d.tolist()[0], rc, rgi, rt, rgw,  rswp, r_mw ]
    plt.stackplot(x, r, labels=['Desal', 'Cachuma', 'Gibraltar', 'Tunnel', 'Groundwater', 'State Water Project', 'Market Water'])
    plt.legend(loc = 'lower left')
    plt.xlabel('Time')
    plt.ylabel('Demand [AFd]')
    plt.title('Water Demand')
    # plt.savefig('../figures/demand_carry.png',  dpi=100)
    plt.show()

                  
    
    sw_c, sw_g, sw_t, gw_c, dw_c, swp_c, mw_c, distr_c = sim.cost_traj(rc, rgi, rgw, sgw, rd, rswp, r_mw, rt)
    c_d = mat.repmat(dw_c, 1, H)
    # dis_c = mat.repmat(distr_c, 1, H)
    c = [c_d.tolist()[0], sw_c, sw_g, sw_t, gw_c,  swp_c, mw_c]
    plt.stackplot(x, c, labels=['Desal',  'Cachuma', 'Gibraltar', 'Tunnel', 'Groundwater', 'State Water Project', 'Market Water'])
    plt.legend(loc = 'upper left')
    plt.title('Cost')
    plt.xlabel('Time')
    plt.ylabel('Cost [$]')

    # plt.savefig('../figures/cost_carry.png',  dpi=100)
    plt.show()
    
    

    
    # fig, axs = plt.subplots(2,2)
    # axs[0,0].plot(sim.cachuma.inflow)
    # fig.savefig('../figures/inflow.png',  dpi=100)
    
    
    with open('../traj/sc' + s + '.txt', 'w') as filehandle:
        for listitem in sc:
            filehandle.write('%s\n' % listitem)
    
    with open('../traj/rc' + s + '.txt', 'w') as filehandle:
        for listitem in rc:
            filehandle.write('%s\n' % listitem)
            
    with open('../traj/sgi' + s + '.txt', 'w') as filehandle:
        for listitem in sgi:
            filehandle.write('%s\n' % listitem)
    
    with open('../traj/rgi' + s + '.txt', 'w') as filehandle:
        for listitem in rgi:
            filehandle.write('%s\n' % listitem)
    
    with open('../traj/sgw' + s + '.txt', 'w') as filehandle:
        for listitem in sgw:
            filehandle.write('%s\n' % listitem)
            
    with open('../traj/rgw' + s + '.txt', 'w') as filehandle:
        for listitem in rgw:
            filehandle.write('%s\n' % listitem)

    with open('../traj/sswp' + s + '.txt', 'w') as filehandle:
        for listitem in sswp:
            filehandle.write('%s\n' % listitem)
    
    with open('../traj/rswp' + s + '.txt', 'w') as filehandle:
        for listitem in rswp:
            filehandle.write('%s\n' % listitem)
            
    with open('../traj/r_mw' + s + '.txt', 'w') as filehandle:
        for listitem in r_mw:
            filehandle.write('%s\n' % listitem)
    
    
    with open('../traj/sw_c' + s + '.txt', 'w') as filehandle:
        for listitem in sw_c:
            filehandle.write('%s\n' % listitem)

    with open('../traj/sw_g' + s + '.txt', 'w') as filehandle:
        for listitem in sw_g:
            filehandle.write('%s\n' % listitem)
    
    with open('../traj/gw_c' + s + '.txt', 'w') as filehandle:
        for listitem in gw_c:
            filehandle.write('%s\n' % listitem)
            
    with open('../traj/swp_c' + s + '.txt', 'w') as filehandle:
        for listitem in swp_c:
            filehandle.write('%s\n' % listitem)
            
    with open('../traj/mw_c' + s + '.txt', 'w') as filehandle:
        for listitem in mw_c:
            filehandle.write('%s\n' % listitem)
    
    
    
    
    
    
    
