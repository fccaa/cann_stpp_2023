# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:54:36 2023

@author: ZHAO Huilin
"""
'''In the Presence of a Continuously Moving Stimulus'''

import numpy as np
import scipy.stats as stats
import scipy.integrate as spint
import matplotlib.pyplot as plt

''' construct the class for model'''

class cann_model:
    
    # define the range of perferred stimuli 
    z_min = - np.pi;              
    z_range = 2.0 * np.pi;    
      
    # define the time scale  
    tau = 10.0 
    tau1 = 50.0
    tau2 = 500.0      
    
    # function for periodic boundary condition   
    def dist(self, c): 
        tmp = np.remainder(c, self.z_range)    
        
        # routine for numbers   
        if isinstance(tmp, (int, float)): 
            if tmp > (0.5 * self.z_range):
                return (tmp - self.z_range); 
            return tmp; 
        
        # routine for numpy arraies  
        for tmp_1 in np.nditer(tmp, op_flags=['readwrite']):   
            if tmp_1 > (0.5 * self.z_range):
                tmp_1[...] = tmp_1 - self.z_range;
        
        return tmp;
    
    # initiation
    def __init__(self, argument):
        self.k = argument.get("k", 0.5);              # rescaled inhibition             
        self.a = argument.get("a", 0.5);              # range of excitatory connection 
        self.N = argument.get("N", 200);              # number of units / neurons   
        self.alpha = argument.get("alpha", 5);        # range of excitatory connection 
        self.beta = argument.get("beta", 5);          # number of units / neurons
        self.sti_v = argument.get("sti_v", 0.001);    # number of stimulus velocity (rad/ms)
        self.A=argument.get("A",0.1);                 # size of stimulus
        self.dx = self.z_range / self.N               # separation between neurons
        
        # define perferred stimulus for each neuron
        self.x = (np.arange(0,self.N,1)+0.5) * self.dx + self.z_min;  # [-pi,pi]
        
        # calculate the excitatory couple for each pair of neurons
        self.Jxx = np.zeros((self.N, self.N)); 
        for i in range(self.Jxx.shape[0]):
            for j in range(self.Jxx.shape[1]):
                self.Jxx[i][j] = \
                np.exp(-0.5 * np.square(self.dist(self.x[i] \
                                                  - self.x[j]) / self.a)) \
                / (np.sqrt(2*np.pi) * self.a); # J0=1
                
        self.y = np.zeros((self.N)*3);       # initialize neuronal inputs         
        self.r = np.zeros((self.N));         # initialize neuronal activities       
        self.input = np.zeros((self.N));     # initialize the external input  
    
    # function for setting external input for each neuron
    def set_input(self, A, z0):
        self.input = A * np.exp(-0.25 * np.square(self.dist(self.x - z0) / self.a));  
    
    # function for calculation of neuronal activity of each neuron
    def cal_r_or_u(self, y):
        u = y[0:self.N]
        u0 = 0.5 * (u + np.abs(u));                                            
        r = np.square(u0);
        B = 1.0 + 0.125 * self.k * np.sum(r) * self.dx / (np.sqrt(2*np.pi) * self.a);
        r = r / B;
        return r
    
    # calculate the centre of mass of u
    def cm_of_u(self, y):                                                      
        u = y[0:self.N]                                                        
        max_i = u.argmax()
        cm = np.dot(self.dist(self.x - self.x[max_i]), u) / np.sum(u)          
        cm = cm + self.x[max_i]                                               
        return self.dist(cm)

    # function of r driving S                                                  
    @staticmethod
    def func_S(r):                                                       
        sigma = 2 
        loc = 6.0 
        return stats.norm.cdf(r, loc=loc, scale=sigma)
    
    # function of r driving Q
    @staticmethod
    def func_Q(r):
        sigma = 0.5                                                            
        mode = 1.0
        mu = np.log(mode) + sigma**2
        return stats.lognorm.pdf(r, sigma, scale=np.exp(mu))                 
    
    # function for calculation of derivatives
    def get_dydt(self, t, y):
        u = y[self.N*0:self.N*1]
        S = y[self.N*1:self.N*2]
        Q = y[self.N*2:self.N*3]
        
        self.set_input(A=self.A, z0=self.dist(self.sti_v *t ))
        
        r = self.cal_r_or_u(u)
        
        I_tot=np.dot(self.Jxx, r)*self.dx + self.input  
        
        dudt = -u + (1+S)* (I_tot);          
        
        dSdt = -S / self.tau1 + self.alpha * Q * self.func_S(r)
        
        dQdt = -Q / self.tau2 - self.alpha * Q * self.func_S(r)+ self.beta * (1-Q) * self.func_Q(I_tot)
        
        dudt = dudt / self.tau;
        
        return np.concatenate((dudt, dSdt, dQdt))

'''Simulation'''

arg = {}

arg["k"] = 0.5
arg["a"] = 0.5
arg["N"] = 200
arg["A"] = 2
arg["sti_v"]=0.006 # delayed:0.006; perfect:0.00425; anticipatory:0.003

# ==============================with STPP======================================
arg["alpha"] = 0.02
arg["beta"]=0.1

# construct a CANN object
cann = cann_model(arg)

# setting up an external input according to the input parameter
cann.sti_v=0
cann.set_input(arg["A"], 0)
out = spint.solve_ivp(cann.get_dydt, (0, 100), cann.y, method="RK45");
cann.y = out.y[:,-1]

# take a initial snapshot
snapshots=np.array([cann.y])
snapshots_cm = [cann.cm_of_u(cann.y)]
snapshots_z0 = [0]

# set the speed
cann.sti_v = arg["sti_v"]

time_unit=10
# run the simulation and take snapshots every 10 taus
for t in range(0,3000,time_unit):
    # decide the period of this step
    t0 = t
    t1 = t + time_unit
    # run the simulation and update the state in the CANN object
    out = spint.solve_ivp(cann.get_dydt, (t0, t1), cann.y, method="RK45");
    cann.y = out.y[:,-1]
    # store the snapshot
    snapshots = np.append(snapshots, [cann.y.transpose()], axis=0)
    snapshots_cm.append(cann.cm_of_u(cann.y)) 
    snapshots_z0.append(cann.dist(arg["sti_v"]*t1)) # moving stimulus

# ==============================without STPP===================================
arg["alpha"] = 0.0
arg["beta"]=0.0

# construct a CANN object
cann0 = cann_model(arg)

# setting up an external input according to the input parameter
cann0.sti_v=0
cann0.set_input(arg["A"], 0)
out = spint.solve_ivp(cann0.get_dydt, (0, 100), cann0.y, method="RK45");
cann0.y = out.y[:,-1]

# take a initial snapshot
snapshots0=np.array([cann0.y])
snapshots0_cm = [cann0.cm_of_u(cann0.y)]
snapshots0_z0 = [0]

# set the speed
cann0.sti_v = arg["sti_v"]

time_unit=10
# run the simulation and take snapshots every 10 taus
for t in range(0,3000,time_unit):
    # decide the period of this step
    t0 = t
    t1 = t + time_unit
    # run the simulation and update the state in the CANN object
    out = spint.solve_ivp(cann0.get_dydt, (t0, t1), cann0.y, method="RK45");
    cann0.y = out.y[:,-1]
    # store the snapshot
    snapshots0 = np.append(snapshots0, [cann0.y.transpose()], axis=0)
    snapshots0_cm.append(cann0.cm_of_u(cann0.y)) 
    snapshots0_z0.append(cann0.dist(arg["sti_v"]*t1)) # moving stimulus
    
#%% plot figure 4A/B/C and figure 8

# ===========================figure 4A/B/C=====================================
plt.figure(figsize=(4.0,4.8))
plt.plot(np.arange(len(snapshots_cm))*time_unit,snapshots_z0,linewidth=2.5,label="Moving Stimulus",color="darkorange")
plt.plot(np.arange(len(snapshots_cm))*time_unit,snapshots_cm,linewidth=2.5,label="With STPP",color="steelblue")
plt.plot(np.arange(len(snapshots0_cm))*time_unit,snapshots0_cm,linewidth=2.5,label="No STPP",color="darkgreen")

plt.legend(loc="upper left",frameon=False,fontsize=17,numpoints=4)

plt.xlim((0,500))
plt.ylim((-0.1,3.1))
plt.tick_params(width=2.0, labelsize=17)
plt.xlabel(r"$t$ (ms)",fontsize=18)
plt.ylabel(r"$z(t)$ (rad)",fontsize=18)

bwith = 2.0 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

plt.tight_layout(pad=1)

# ================================figure 8=====================================
'''set:
   arg["sti_v"]=0.003
   arg["alpha"] = 0.02
   arg["beta"]=0.1
   and run the simulation code.
'''
fig, ax = plt.subplots(4,1,figsize=(3.5,5))

ls=15
ts=14
bwith = 1.5
for a in range(4):
    ax[a].spines['bottom'].set_linewidth(bwith)
    ax[a].spines['left'].set_linewidth(bwith)
    ax[a].spines['top'].set_linewidth(bwith)
    ax[a].spines['right'].set_linewidth(bwith) 
    ax[a].tick_params(width=1.5, labelsize=ts)
      
ax[3].set_xlabel(r'$x$',fontsize=ls)

ax[0].set_ylabel(r"$I^{\mathrm{ext}}(x,t)$",fontsize=ls)
ax[1].set_ylabel(r"$u(x,t)$",fontsize=ls)
ax[2].set_ylabel(r"$S(x,t)$",fontsize=ls)
ax[3].set_ylabel(r"$Q(x,t)$",fontsize=ls)

ax[0].set_xlim(xmax=np.pi, xmin=-np.pi)
ax[0].set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
ax[0].set_xticklabels([])

ax[1].set_xlim(xmax=np.pi, xmin=-np.pi)
ax[1].set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
ax[1].set_xticklabels([])

ax[2].set_xlim(xmax=np.pi, xmin=-np.pi)
ax[2].set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
ax[2].set_xticklabels([])

ax[3].set_xlim(xmax=np.pi, xmin=-np.pi)
ax[3].set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
ax[3].set_xticklabels([r'$-\pi$',r'$-\pi/2$','0',r'$\pi/2$',r'$\pi$'],fontsize=ts)

fig.tight_layout(pad=0.5)

ax[0].set_ylim((-0.2,2.5))
ax[0].set_yticks([0,1,2])

ax[1].set_ylim((-0.2,15))
ax[1].set_yticks([0,5,10,15])

ax[2].set_ylim((-0.01,0.3))
ax[2].set_yticks([0,0.1,0.2,0.3])

ax[3].set_ylim((-0.05,1.1))
ax[3].set_yticks([0,0.5,1.0])

frame=0 # change from 0,10,20,40,120 to plot frames at specific time points
z0=cann.dist(cann.sti_v *frame*10 )
I=cann.A * np.exp(-0.25 * np.square(cann.dist(cann.x - z0) / cann.a))

l1, = ax[0].plot(cann.x,I ,"darkorange",lw=2)
l2, = ax[1].plot(cann.x, snapshots[frame,:cann.N*1], "steelblue",lw=2)
l3, = ax[2].plot(cann.x, snapshots[frame,cann.N*1:cann.N*2],"orangered",lw=2)
l4, = ax[3].plot(cann.x, snapshots[frame,cann.N*2:cann.N*3],"purple",lw=2)
title_out=r"$t=$"+str(frame*10)+" ms"
ax[0].set_title(title_out,fontsize=ls)
plt.tight_layout(pad=1)
