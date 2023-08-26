# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 11:05:58 2023

@author: ZHAO Huilin
"""
'''Intrinsic Dynamics'''

import numpy as np
import scipy.stats as stats
import scipy.integrate as spint
import matplotlib.pyplot as plt
from interval import Interval
from joblib import Parallel, delayed

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

def cal_velocity(cann,snapshots_cm,time_unit):
    diff=[]
    j=-1
    for i in range(10):
        diff.append(cann.dist(snapshots_cm[j-i]-snapshots_cm[j-i-1]))
    
    flag=[]
    interval=Interval(-1e-4, 1e-4)
    for i in range(9):
        if (diff[i]-diff[i+1]) in interval:
            flag.append(1)
        else: flag.append(0)
    if np.sum(flag)==len(flag):
        print("Velocity is "+str(np.mean(diff)/time_unit))
        return np.mean(diff)/time_unit      
    else: 
        print("Not Constant Speed")
        return -1

def intrinsic(alpha,beta,arg):
    arg["alpha"] = alpha
    arg["beta"]=beta
    
    cann = cann_model(arg)    
    if arg["k"] < 1.0:
        cann.set_input(np.sqrt(32.0)/arg["k"], 0) 
    else:
        cann.set_input(np.sqrt(32.0), 0)
    cann.y[0:cann.N] = cann.input
    
    cann.set_input(arg["A"],0)
    out = spint.solve_ivp(cann.get_dydt, (0, 100), cann.y, method="RK45")
    cann.y = out.y[:,-1] 
    
    cann.set_input(0, 0)
    for t in range(100):
        out = spint.solve_ivp(cann.get_dydt, (0, 10), cann.y, method="RK45");
        cann.y = out.y[:,-1]
        cann.y[0:cann.N] = np.roll(cann.y[0:cann.N], 1)
 
    cann.set_input(0, 0)
      
    out = spint.solve_ivp(cann.get_dydt, (0, 1000), cann.y, method="RK45");

    # take a initial snapshot
    snapshots = np.array([cann.y])
    snapshots_cm = [cann.cm_of_u(cann.y)]
    
    time_unit=1
    for t in range(0,50,time_unit):
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
    v=cal_velocity(cann,snapshots_cm,time_unit)
    
    return v

'''Simulation'''

arg = {}

arg["k"] = 0.5
arg["a"] = 0.5
arg["N"] = 200
arg["A"] = 2.0
arg["z0"] = 0

beta_range=np.arange(0.00,0.200001,0.004)
alpha_range=np.arange(0.00,0.200001,0.004)

vel_dict={"alpha_range": alpha_range,"beta_range":beta_range,"arg":arg}

vel_mat=np.ones((len(alpha_range),len(beta_range)))
for i in range(len(alpha_range)):
    vel_mat[i,:]=Parallel(n_jobs=-1)(delayed(intrinsic)(alpha_range[i], beta_i,arg) for beta_i in beta_range)
    print("alpha: "+str(alpha_range[i])+" finish")
vel_dict["vel_mat"]=vel_mat
np.save(".../intrinsic_speed_alpha_{:.2f}-{:.2f}_beta_{:.2f}-{:.2f}.npy"\
        .format(alpha_range[0],alpha_range[-1],beta_range[0],beta_range[-1]),vel_dict)


#%% plot figure 6A&B, figure S1
'''For figure S1, should change the value of tau2 and run the upper code to get
the intrinsic speeds matrixes, and then plot the contour map.'''
# ====================contour map of intrinsic speeds==========================
# figure 6A/S1

labelfs=18
ticksfs=17

alpha_range=vel_dict["alpha_range"]
beta_range=vel_dict["beta_range"]

plt.figure(figsize=(6.2,6))
plt.tick_params(width=2.0, labelsize=ticksfs)

C=plt.contour(beta_range,alpha_range,vel_mat,[0.000001,0.003,0.004,0.005,0.006,0.007,0.008],\
              colors=["orangered","darkorange","limegreen","deepskyblue","royalblue","darkviolet"],linewidths=2.5)  
manual_location=[(0.028,0.015),(0.098,0.01),(0.128,0.020),\
                 (0.135,0.026),(0.140,0.046),(0.146,0.080),\
                 (0.150,0.144)] # the label of 0.00001 needs to be adjusted manually in Adobe Illustrator
plt.clabel(C, inline=True, fontsize=ticksfs,manual=manual_location)

plt.xticks([0.00,0.05,0.10,0.15,0.20])
plt.yticks([0.00,0.05,0.10,0.15,0.20])
plt.xlabel(r"$\beta$",fontsize=labelfs+1)
plt.ylabel(r"$\alpha$",fontsize=labelfs+1)
plt.title(r"$V_\mathrm{int}$ (rad/ms)",fontsize=labelfs)

bwith = 2.0 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

plt.tight_layout(pad=1)

# =================relationship of vint and alpha/beta=========================
# figure 6B

alpha_range=vel_dict["alpha_range"]
beta_range=vel_dict["beta_range"]

labelfs=14
ticksfs=13

# fixed alpha
fig,ax=plt.subplots(figsize=(3.0,2.4))
ax.tick_params(width=2.0, labelsize=ticksfs)
for alpha in np.arange(0.04,0.17,0.04):
    a_idx=int(np.where(alpha_range==alpha)[0])
    ax.plot(beta_range,vel_mat[a_idx,:],label=r"$\alpha = $"+"{:.2f}".format(alpha),linewidth=2.5)

ax.set_xticks([0,0.05,0.10,0.15,0.2])
ax.set_ylabel(r"$V_\mathrm{int}$"+r" ($\times 10^{-3}$rad/ms)",fontsize=labelfs)
ax.set_yticks([0.00,0.002,0.004,0.006,0.008])
ax.set_yticklabels([0,2,4,6,8])
ax.set_xlabel(r"$\beta$",fontsize=labelfs+1)
ax.legend(loc="lower right",numpoints=4,frameon=False,fontsize=ticksfs)

bwith = 2.0 
ax1 = plt.gca()
ax1.spines['bottom'].set_linewidth(bwith)
ax1.spines['left'].set_linewidth(bwith)
ax1.spines['top'].set_linewidth(bwith)
ax1.spines['right'].set_linewidth(bwith)

plt.tight_layout(pad=1)

# fixed beta
fig,ax=plt.subplots(figsize=(3.0,2.4))
ax.tick_params(width=2.0, labelsize=ticksfs)
for beta in np.arange(0.04,0.17,0.04):
    b_idx=int(np.where(beta_range==beta)[0])
    plt.plot(alpha_range,vel_mat[:,b_idx],label=r"$\beta = $"+"{:.2f}".format(beta),linewidth=2.5)

ax.set_xticks([0,0.05,0.10,0.15,0.2])
ax.set_ylabel(r"$V_\mathrm{int}$"+r" ($\times 10^{-3}$rad/ms)",fontsize=labelfs)
ax.set_yticks([0.00,0.002,0.004,0.006,0.008])
ax.set_yticklabels([0,2,4,6,8])
ax.set_xlabel(r"$\alpha$",fontsize=labelfs+1)
ax.legend(loc="lower right",numpoints=4,frameon=False,fontsize=ticksfs)

bwith = 2.0 
ax1 = plt.gca()
ax1.spines['bottom'].set_linewidth(bwith)
ax1.spines['left'].set_linewidth(bwith)
ax1.spines['top'].set_linewidth(bwith)
ax1.spines['right'].set_linewidth(bwith)

plt.tight_layout(pad=1)

