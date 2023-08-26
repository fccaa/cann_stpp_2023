# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 21:24:21 2023

@author: ZHAO Huilin
"""
'''Anticipatory Tracking'''

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
    
def moving_sim(arg:dict):
    # construct a CANN object
    cann = cann_model(arg)

    # setting up an external input according to the inputted parameter
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
    if cann.sti_v<0.001: # need longer time to converge
        max_time=10000
    else:
        max_time=5000
    # run the simulation and take snapshots every 10 taus
    for t in range(0,max_time,time_unit):
        # decide the period of this step
        t0 = t
        t1 = t + time_unit
        # run the simulation and update the state in the CANN object
        out = spint.solve_ivp(cann.get_dydt, (t0, t1), cann.y, method="RK45");
        cann.y = out.y[:,-1]
        # store the snapshot
        snapshots = np.append(snapshots, [cann.y.transpose()], axis=0)
        cm=cann.cm_of_u(cann.y)
        snapshots_cm.append(cm)
        z0=cann.dist(arg["sti_v"]*t1)
        snapshots_z0.append(z0) # moving stimulus
        
    return (snapshots[-1],snapshots_cm[-1],snapshots_z0[-1]),cann

'''Simulation'''

arg = {}

arg["k"] = 0.5
arg["a"] = 0.5
arg["N"] = 200
arg["sti_v"]=0.01
# adjustable
arg["A"] = 2
arg["alpha"] = 0.02
arg["beta"]=0.10

v_range=np.arange(0.0,0.008001,0.0002)
     
snapshots=[]
for v in v_range:
    arg["sti_v"] = v
    sn,cann=moving_sim(arg)
    snapshots.append(sn)
    print("v: {:.4f} cm: {:.4f} z0: {:.4f}".format(v,sn[1],sn[2]))

d=[cann.dist(snapshots[i][1]-snapshots[i][2]) for i in range(len(snapshots))]       
mov_snap_dict={"v_range": v_range, "snapshots": snapshots,"displacement":d,"arg":arg}
np.save(".../alpha_{:.3f}_beta_{:.3f}_A_{:.1f}_moving_tracking.npy".format(arg["alpha"],arg["beta"],arg['A']),mov_snap_dict)

#%% plot figure 5 B/C

'''
Run the upper cell for different combinations of alpha, beta and A.
For example, let's name the results as follow:
    alpha  beta  A
m1  0.02   0.10  2
m2  0.06   0.06  3
m3  0.02   0.01  2
m4  0.00   0.00  2
'''
m1=np.load(".../alpha_0.020_beta_0.100_A_2.0_moving_tracking.npy",allow_pickle=True).item()
m2=np.load(".../alpha_0.060_beta_0.060_A_3.0_moving_tracking.npy",allow_pickle=True).item()
m3=np.load(".../alpha_0.020_beta_0.010_A_2.0_moving_tracking.npy",allow_pickle=True).item()
m4=np.load(".../alpha_0.000_beta_0.000_A_2.0_moving_tracking.npy",allow_pickle=True).item()

plt.figure(figsize=(3.8,3.8))
tickfs=13
labelfs=14
plt.plot(m1["v_range"],np.zeros(len(m1["v_range"])),color="grey",linestyle="--",linewidth=2)

plt.plot(m1["v_range"],m1["displacement"],label=(r"$\alpha=0.02$"+" "+r"$\beta=0.10$"+" "+r"$A=2$"),linewidth=2.5)
plt.plot(m2["v_range"],m2["displacement"],label=(r"$\alpha=0.06$"+" "+r"$\beta=0.06$"+" "+r"$A=3$"),linewidth=2.5)
plt.plot(m3["v_range"],m3["displacement"],label=(r"$\alpha=0.02$"+" "+r"$\beta=0.01$"+" "+r"$A=2$"),linewidth=2.5)
plt.plot(m4["v_range"],m4["displacement"],label=(r"$\alpha=0.00$"+" "+r"$\beta=0.00$"+" "+r"$A=2$"),linewidth=2.5)
plt.xlim((-0.0000,0.0065))
plt.ylim((-0.55,0.14))
plt.xlabel(r"$v_{\mathrm{ext}}$ (rad/ms)",fontsize=labelfs)
plt.ylabel("$s$ (rad)",fontsize=labelfs)
plt.legend(loc="lower left",numpoints=4,fontsize=tickfs,frameon=False)
plt.tick_params(width=2.0, labelsize=tickfs)

bwith = 2.0
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

plt.tight_layout(pad=1)


