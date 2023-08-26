# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:42:58 2023

@author: ZHAO Huilin
"""
'''In the Presence of an Abruptly Changing Stimulus'''

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
        self.k = argument.get("k", 0.5);       # rescaled inhibition             
        self.a = argument.get("a", 0.5);       # range of excitatory connection 
        self.N = argument.get("N", 200);       # number of units / neurons   
        self.alpha = argument.get("alpha", 5); # range of excitatory connection 
        self.beta = argument.get("beta", 5);   # number of units / neurons
        self.A=argument.get("A",0.1);          # size of stimulus
        self.dx = self.z_range / self.N        # separation between neurons
        
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
arg["A"] = 3.0
arg["z0"] = 1

# ==============================with STPP======================================
arg["alpha"] = 0.02 
arg["beta"]=0.1 

cann = cann_model(arg)

cann.set_input(arg["A"],0)
out = spint.solve_ivp(cann.get_dydt, (0, 100), cann.y, method="RK45")
cann.y = out.y[:,-1] 

# take an initial snapshot
snapshots = np.array([cann.y]) 
snapshots_cm = [cann.cm_of_u(cann.y)]

cann.set_input(arg["A"],arg["z0"])

time_unit=10
for t in range(0, 1000,time_unit): 
    # decide the period of this step
    t0 = t
    t1 = t + time_unit
    # run the simulation and update the state in the CANN object
    out = spint.solve_ivp(cann.get_dydt, (t0, t1), cann.y, method="RK45");
    cann.y = out.y[:,-1]
    # store the snapshot
    snapshots = np.append(snapshots, [cann.y.transpose()], axis=0) 
    snapshots_cm.append(cann.cm_of_u(cann.y))
snapshots_cm=np.array(snapshots_cm)

# ================================without STPP=================================
arg["alpha"] = 0.0
arg["beta"]=0.0

cann0 = cann_model(arg)

if arg["k"] < 1.0:
    cann0.set_input(np.sqrt(32.0)/arg["k"], 0)
else:
    cann0.set_input(np.sqrt(32.0), 0)
cann0.y[0:cann0.N] = cann0.input 

cann0.set_input(arg["A"],0)
out = spint.solve_ivp(cann0.get_dydt, (0, 100), cann0.y, method="RK45")
cann0.y = out.y[:,-1] 

# take an initial snapshot
snapshots0 = np.array([cann0.y]) 
snapshots0_cm = [cann0.cm_of_u(cann0.y)]

cann0.set_input(arg["A"],arg["z0"])

time_unit=10
for t in range(0, 1000,time_unit): 
    # decide the period of this step
    t0 = t
    t1 = t + time_unit
    # run the simulation and update the state in the CANN object
    out = spint.solve_ivp(cann0.get_dydt, (t0, t1), cann0.y, method="RK45");
    cann0.y = out.y[:,-1]
    # store the snapshot
    snapshots0 = np.append(snapshots0, [cann0.y.transpose()], axis=0) 
    snapshots0_cm.append(cann0.cm_of_u(cann0.y))
snapshots0_cm=np.array(snapshots0_cm)

#%% plot figure 2&3

# ==================plot figure 2A: function f_Q & f_S=========================
x=np.arange(0,10,0.1)
fig,ax=plt.subplots(2,1,figsize=(3,4))

ax[0].plot(x, cann.func_Q(x),color="steelblue",linewidth=2.5,linestyle='-',label=r"$func_Q$")
ax[0].tick_params(width=2.0, labelsize=14)
ax[0].set_xticks(np.arange(0,11,2))
ax[0].set_xticklabels([])
ax[0].set_yticks(np.arange(0,0.7,0.2))
ax[0].set_ylabel(r"$f_Q (I^{\mathrm{tot}})$",fontsize=15)


ax[1].plot(x, cann.func_S(x),color="darkorange",linewidth=2.5,linestyle='-',label=r"$func_S$")
ax[1].tick_params(width=2.0, labelsize=14)
ax[1].set_xticks(np.arange(0,11,2))
ax[1].set_yticks(np.arange(0,1.1,0.5))
ax[1].set_xlabel(r"$I^{\mathrm{tot}}/r$",fontsize=15)
ax[1].set_ylabel(r"$f_S (r)$",fontsize=15)

bwith = 2.0
for i in range(2):
    ax[i].spines['bottom'].set_linewidth(bwith)
    ax[i].spines['left'].set_linewidth(bwith)
    ax[i].spines['top'].set_linewidth(bwith)
    ax[i].spines['right'].set_linewidth(bwith)
    
plt.tight_layout(pad=1)

# ========================plot figure 2B: u(x,t)===============================
plt.figure(figsize=(3.5,4))
plt.plot(cann.x, snapshots[0,:cann.N],color="darkgreen",linewidth=2.5,linestyle='-')
plt.xticks(np.arange(-3,4,1))
plt.yticks(np.arange(0,16,5))
plt.tick_params(width=2.0, labelsize=14)
plt.xlabel(r"$x$",fontsize=15)
plt.ylabel(r"$u(x,t)$",fontsize=15)

bwith = 2.0 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

plt.tight_layout(pad=1)

# ======================plot figure 2C: Q(x,t)&S(x,t)==========================
fig,ax1=plt.subplots(1,1,figsize=(4.8,3.9))
plt.tick_params(width=2.0, labelsize=14)
# Q(x,t)
l1=ax1.plot(cann.x, snapshots[0,cann.N*2:cann.N*3 ],color="orangered",linewidth=2.5,linestyle='-',\
            label=r"$Q(x,t)$")
ax1.set_xticks(np.arange(-3,4,1))
ax1.set_xticklabels(np.arange(-3,4,1),fontsize=14)
ax1.set_xlabel(r"$x$",fontsize=15)
ax1.set_ylim((-0.051,1.15))
ax1.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ax1.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0],fontsize=14)
ax1.set_ylabel(r"$Q(x,t)$",fontsize=15)
# S(x,t)
ax2=ax1.twinx()
l2=ax2.plot(cann.x, snapshots[0,cann.N*1:cann.N*2 ],color="purple",linewidth=2.5,linestyle='-',\
            label=r"$S(x,t)$")
ax2.set_ylim((-0.00148,0.0335))
ax2.set_yticks(np.arange(0.00,0.031,0.01))
ax2.set_yticklabels(np.arange(0.00,0.031,0.01),fontsize=14)
ax2.set_ylabel(r"$S(x,t)$",fontsize=15)

bwith = 2.0
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

line = l1+l2
labs = [l.get_label() for l in line]
ax1.legend(line,labs,loc="upper left",ncol=2,fontsize=14,frameon=False)
# ax1.legend(loc="upper left",frameon=False,fontsize=13)
# ax2.legend(loc="upper right",frameon=False,fontsize=13)
plt.tight_layout(pad=1)

# ========================plot 3A: snapshot of u(x,t)==========================
plt.figure(figsize=(4,3))
color_list = plt.cm.Blues(np.linspace(0.2, 0.8, 5))

plt.plot(cann.x, snapshots[0,:cann.N],color=color_list[0],linewidth=2,linestyle='-')
    
for i in np.arange(1,4):
    plt.plot(cann.x,snapshots[i*4,:cann.N],color=color_list[i],linewidth=2,linestyle="--")
    
plt.plot(cann.x, snapshots[-1,:cann.N],color=color_list[-1],linewidth=2,linestyle='-')

plt.ylim((-0.3,18.3))
plt.xticks(np.arange(-3,4,1),fontsize=13)
plt.yticks(np.arange(0,16,5),fontsize=13)
plt.tick_params(width=1.5, labelsize=13)
plt.xlabel(r"$x$",fontsize=14)
plt.ylabel(r"$u(x,t)$",fontsize=14)

bwith = 1.5
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

plt.tight_layout(pad=1)

# =====================plot 3B: z(t) of u(x,t) w/ and w/o STPP=================
plt.figure(figsize=(4,3))
plt.plot(np.arange(snapshots_cm.shape[0])*10,snapshots_cm,label="With STPP",color="steelblue",\
         linewidth=2)
plt.plot(np.arange(snapshots0_cm.shape[0])*10,snapshots0_cm,label="Without STPP",color="darkorange",\
         linewidth=2)

plt.xticks(np.arange(0,1001,200),fontsize=13)
plt.yticks(np.arange(0,1.3,0.2),fontsize=13)
plt.tick_params(width=1.5, labelsize=13)
plt.xlabel(r"$t$ (ms)",fontsize=14)
plt.ylabel(r"$z(t)$ (rad)",fontsize=14)

bwith = 1.5 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

plt.legend(loc='lower right',fontsize=13,frameon=False)
plt.tight_layout(pad=1)

