# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 12:20:35 2023

@author: ZHAO Huilin
"""
'''Translational Stability Analysis'''

import numpy as np
import scipy.stats as stats
import scipy.integrate as spint
import matplotlib.pyplot as plt
import matplotlib as mpl
from joblib import Parallel, delayed

# Class for the model

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
        self.k = argument.get("k", 0.5);         # rescaled inhibition
        self.a = argument.get("a", 0.5);         # range of excitatory connection
        self.N = argument.get("N", 200);         # number of units / neurons
        self.alpha = argument.get("alpha", 5);   # range of excitatory connection
        self.beta = argument.get("beta", 5);     # number of units / neurons
        self.A = argument.get("A", 0);           # size of stimulus 
        self.dx = self.z_range / self.N          # separation between neurons
        
        # define perferred stimuli for each neuron
        self.x = (np.arange(0,self.N,1)+0.5) * self.dx + self.z_min;
        
        # calculate the excitatory couple for each pair of neurons
        self.Jxx = np.zeros((self.N, self.N));
        for i in range(self.Jxx.shape[0]):
            for j in range(self.Jxx.shape[1]):
                self.Jxx[i][j] =                 np.exp(-0.5 * np.square(self.dist(self.x[i]                                                   - self.x[j]) / self.a))                 / (np.sqrt(2*np.pi) * self.a);
                
        self.y = np.zeros((self.N)*3);      # initialize neuronal inputs
        self.r = np.zeros((self.N));        # initialize neuronal activities
        self.input = np.zeros((self.N));    # initialial the external input
    
    # function for setting external iput for each neuron
    def set_input(self, A, z0):
        self.input = A * np.exp(-0.25 * np.square(self.dist(self.x - z0) / self.a));
    
    # function for calculation of neuronal activity of each neuron
    def cal_r_or_u(self, y):
        u = y[0:self.N]
        u0 = 0.5 * (u + np.abs(u));
        r = np.square(u0);
        B = 1.0 + 0.125 * self.k * np.sum(r) * self.dx / (np.sqrt(2*np.pi) * self.a);
        r = r / B;
        return r;
    
    # calculate the centre of mass of u
    def cm_of_u(self, y):
        u = y[0:self.N]
        max_i = u.argmax()
        cm = np.dot(self.dist(self.x - self.x[max_i]), u) / np.sum(u)
        cm = cm + self.x[max_i]
        return self.dist(cm);
    
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
        mu = np.log(mode) + sigma ** 2
        return stats.lognorm.pdf(r, sigma, scale=np.exp(mu))
    
    # function for calculation of derivatives
    def get_dydt(self, t, y):
        u = y[self.N*0:self.N*1]
        S = y[self.N*1:self.N*2]
        Q = y[self.N*2:self.N*3]
        
        u = 0.5 * (u + u[::-1])
       
        r = self.cal_r_or_u(u)
        
        I_tot = np.dot(self.Jxx, r) * self.dx + self.input
        
        dudt = -u + (1+S)*(I_tot);
        
        dSdt = -S / self.tau1 + self.alpha * Q * self.func_S(r)
        
        dQdt = -Q / self.tau2 - self.alpha * Q * self.func_S(r) + self.beta * (1-Q) * self.func_Q(I_tot)
        
        dudt = dudt / self.tau;
        
        return np.concatenate((dudt, dSdt, dQdt))     

def one_sim(alpha = 1e-20, beta = 1e-20):

    # Define parameters
    arg = {}

    arg["k"] = 0.5
    arg["a"] = 0.5
    arg["N"] = 200
    arg["A"] = 2
    arg["z0"] = 1

    arg["alpha"] = alpha
    arg["beta"] = beta

    # construct a CANN object
    cann = cann_model(arg)

    # setting up an initial condition of neuronal inputs 
    # so that tracking can be reasonable for small A and k < 1
    if arg["k"] < 1.0:
        cann.set_input(np.sqrt(32.0)/arg["k"], 0)
    else:
        cann.set_input(np.sqrt(32.0), 0)
    cann.y[0:cann.N] = cann.input

    # setting up an external input according to the inputted parameter
    cann.set_input(arg["A"], 0)

    # run the simulation for 100 tau to excite the network state
    out = spint.solve_ivp(cann.get_dydt, (0, 1000), cann.y, method="RK45");

    # update the network state in the CANN object
    cann.y = out.y[:,-1]

    # setting up an external input according to the inputted parameter
    cann.set_input(0, 0)

    # run the simulation for 100 tau to a stationary state
    out = spint.solve_ivp(cann.get_dydt, (0, 5000), cann.y, method="RK45");

    # update the network state in the CANN object
    cann.y = out.y[:,-1]
    
    u = cann.y[cann.N*0:cann.N*1]
    S = cann.y[cann.N*1:cann.N*2]
    Q = cann.y[cann.N*2:cann.N*3]
    
    x = cann.x
    Jxx = cann.Jxx

    dudx = np.gradient(u, cann.x)
    dSdx = x * S
    dQdx = x * Q

    B = 1 + 0.125 * cann.k * np.trapz(u*u, x) / (np.sqrt(2.*np.pi)*cann.a)

    r = u*u / B

    I_tot = np.array([np.trapz(Jxx[i]*r, x) for i in range(Jxx.shape[1])])
    
    inner_int = np.array([np.trapz(Jxx[i]*u*dudx, x) for i in range(Jxx.shape[1])])

    Muu = (-1.0 + 2.0 * np.trapz(dudx * (1.0+S) * inner_int, x) / (B * np.trapz(dudx*dudx, x))) / cann.tau

    MuS = (np.trapz(dudx*dSdx*I_tot, x) / (np.trapz(dudx*dudx, x))) / cann.tau

    MuQ = 0.0

    h = 0.0001

    dfSdx_of_r = (cann.func_S(r + h) - cann.func_S(r - h)) / (2.0 * h)

    MSu= 2.0 * cann.alpha * np.trapz(dSdx*Q*dfSdx_of_r*u*dudx, x) / (B * np.trapz(dSdx*dSdx, x))

    MSS = -1.0 / cann.tau1

    MSQ = cann.alpha * np.trapz(dSdx*dQdx*cann.func_S(r), x) / (np.trapz(dSdx*dSdx, x))

    inner_int = np.array([ np.trapz(Jxx[i] * u * dudx, x) for i in range(Jxx.shape[1])])

    dfQdx_of_I = (cann.func_Q(I_tot + h) - cann.func_Q(I_tot - h)) / (2.0 * h)

    MQu = - np.trapz( dQdx * (2.0*cann.alpha*Q*dfSdx_of_r*u*dudx/B 
                           - 2.0*cann.beta*(1.0-Q)*dfQdx_of_I*inner_int/B), x
    ) /  (np.trapz(dQdx*dQdx, x))

    MQS = 0.0

    MQQ = -1.0 / cann.tau2 - np.trapz(
        dQdx * (cann.alpha*dQdx*cann.func_S(r) + cann.beta*dQdx*cann.func_Q(I_tot)), x
    )/  (np.trapz(dQdx*dQdx, x))

    M = [[Muu, MuS, MuQ], [MSu, MSS, MSQ], [MQu, MQS, MQQ]]

    evals, _ = np.linalg.eig(M)
        
    return np.max(np.real(evals))

'''Simulation and Calculation'''

alpha_range = np.arange(1e-20, 0.2000001, 0.001)
beta_range = np.arange(1e-20, 0.2000001, 0.001)

all_result = []
for alpha_i in alpha_range:   
    local_result = Parallel(n_jobs=-1)(delayed(one_sim)(alpha_i, beta_i) for beta_i in beta_range)       
    all_result.append(local_result)        
all_result = np.array(all_result)

#%% plot figure 7A/B
# heatmap of maximal eigenvalues

alpha_range = np.arange(1e-20, 0.2000001, 0.001)
beta_range = np.arange(1e-20, 0.2000001, 0.001)

fig = plt.figure(figsize=(8,7.5))
ax = fig.add_subplot()

labelfs=17
ticksfs=16
plt.tick_params(width=2.0, labelsize=ticksfs)

extent=(np.min(beta_range), np.max(beta_range), np.min(alpha_range), np.max(alpha_range))
cdict = {'red':   [(0.0,  0.0, 0.18),
                    (0.001,  1.0, 1.0),
                    (1.0,  1.0, 0.0)],
          'green': [(0.0,  0.0, 0.18),
                    (0.001, 0.9, 1.0),
                    (1.0,  0.18, 0.0)], 
          'blue':  [(0.0,  1.0, 1.0),
                    (0.001,  0.9, 1.0),
                    (1.0,  0.18, 0.0)]}
cmap =  mpl.colors.LinearSegmentedColormap('cmap', cdict, 6000)

heatmap = ax.imshow(all_result, extent=extent, vmax=0.013, vmin=-0.00001, cmap=cmap,origin="lower")
CR = ax.contour(all_result, levels=[0,0.0001,0.001,0.002], extent=extent, colors="black",linewidths=2) 
manual_loc = [(),(),(0.02,0.02),(0.085,0.015)] # the label locations should be adjusted manually in Adobe Illustrator
ax.clabel(CR, CR.levels,fontsize=ticksfs, manual=manual_loc)

ax.set_xlabel(r"$\beta$",fontsize=labelfs+1)
ax.set_ylabel(r"$\alpha$",fontsize=labelfs+1)
ax.set_xticks([0,0.05,0.1,0.15,0.2])
ax.set_xticklabels(["0.00","0.05","0.10","0.15","0.20"],fontsize=ticksfs)
ax.set_yticks([0,0.05,0.1,0.15,0.2])
ax.set_yticklabels(["0.00","0.05","0.10","0.15","0.20"],fontsize=ticksfs)
ax.set_title(r"$\tilde\lambda$",fontsize=labelfs)


cb = plt.colorbar(heatmap,fraction=0.0455)
cb.set_ticks([0,0.003,0.006,0.009,0.012])
cb.ax.tick_params(width=2.0, labelsize=ticksfs-1)
cb.outline.set_linewidth(2)

bwith = 2 

ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

plt.tight_layout(pad=1)

