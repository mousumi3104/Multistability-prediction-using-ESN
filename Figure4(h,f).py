# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 23:26:14 2022

@author: Mousumi
"""

import math
import random
import scipy.io as io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.sparse import rand
import pylab as pl
from scipy.integrate import odeint
#from numpy import random
#import time
# %matplotlib qt5

import matplotlib.pyplot as plt
from matplotlib import ticker


#----------------------------------------------------------------------------------------------------
def model_hidden(states,t,eps,b):

    x=states[0]
    y=states[1]
    z=states[2]
    s=states[3]
            
    dx = np.zeros(states.shape)

    alpha=-0.02
    k=1
    
    dx[0] = y
    dx[1] = z
    dx[2] = -y+3*y**2-x**2-x*z+alpha+eps*s
    dx[3] = -k*s-eps*(z-b)
    
    return dx

#-----------------------------------------------------------------------------------------------------
def solve_model(epsilon,b,x_in,y_in):

    time=200000
    ntrans=100000
    x_data=np.zeros((time-ntrans,dim*len(epsilon)))

    xy_init = np.array([0,0,0,0.1])
    xy_init[:2] = [x_in,y_in]

    t = np.linspace(0,time*0.01,time)
    statess = np.zeros((len(t),len(epsilon)*dim))

    for ii in range(len(epsilon)):

        statess=odeint(model_hidden, xy_init, t, args=(epsilon[ii],b)) 
        x_data[:,ii*(dim):(ii+1)*(dim)]=statess[ntrans:,:]
        
    return x_data


#--------------------------------------------------------------------------------------------------------    
def W_inn(n,dim,W_in_a):
    W_inputt = np.zeros((n, dim+1))
    for i in range(n):
        W_inputt[i, math.floor(i*dim/n)] = (2*np.random.random()-1)*W_in_a

    W_inputt[:, dim] = (2*np.random.random((n))-1)*W_in_a
    return W_inputt

#-------------------------------------------------------------------------------------------------------
def W_ress(k,n,eig_rho):
    # # ER network n*n and its radius is eig_rho
    prob = k/(n-1)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            b = np.random.random()
            if (i != j) and (b < prob):
                W[i, j] = np.random.random()

    rad = max(abs(LA.eigvals(W)))
    W_reservoirr = W*(eig_rho/rad)  

    return W_reservoirr

#--------------------------------------------------------------------------------------------------------
# ESN dynamics
def create_ESN(ESN_par,x_data,par1,W_in,W_res):
    n=ESN_par[0]
    alpha=ESN_par[1]
    beta=ESN_par[2]
    Nt=ESN_par[3]
    Np=ESN_par[4]
    transit=ESN_par[5]
    dim=ESN_par[6]
    tp_dim=len(par1)
    
# training phase

    U = np.zeros((dim, tp_dim*(Nt-transit+1)))  
    R = np.zeros((n, tp_dim*(Nt-transit+1)))  
    m = np.random.randint(0, 2000)
    for ii in range(tp_dim):
        print(par1[ii])
        u_train = x_data[dim*ii:dim*(ii+1), (m-1):(m+Nt)]
        u1_train = np.vstack((u_train, par1[ii]*np.ones((1, Nt+1))))
        
        r1 = np.zeros((n, Nt+1))
        r2 = np.zeros((n, Nt+1))

        for i in range(Nt):
            # print(i)
            r1[:, i+1] = (1-alpha)*r1[:, i]+alpha *np.tanh(np.dot(W_res, r1[:, i])+np.dot(W_in, u1_train[:, i]))
            r2[:, i+1] = r1[:, i+1]
            r2[1::2, i+1] = r1[1::2, i+1]**2

        U[:, (Nt-transit+1)*ii:(Nt-transit+1)*(ii+1)] = u1_train[:dim, transit:(Nt+1)]
        R[:, (Nt-transit+1)*ii:(Nt-transit+1)*(ii+1)] = r2[:, transit:(Nt+1)]

    R_T = np.transpose(R)
    W_out = np.dot(np.dot(U, R_T), np.linalg.inv((np.dot(R, R_T)+beta*np.identity(n))))
    
    return W_out

#------------------------------------------------------------------------------------------------------
# predicting phase to check the efficiency of the machine
# this part is only for optimizing the parameter. For this code it is not needed.

def error_cal(train_par,x_data,W_out,W_res,W_in):
    
    rmse_mean=np.zeros(len(train_par))
    mm = np.random.randint(0, 3000)+60000
    for j in range(len(train_par)):
        u_train = x_data[dim*j:dim*(j+1), mm:mm+Np]
        u_predict = np.vstack((u_train, train_par[j]*np.ones((1, Np))))
        
        r3 = np.zeros((n, Np))
        r4 = np.zeros((n))
        for i in range(Np-1):
            r3[:, i+1] = (1-alpha)*r3[:, i]+alpha*np.tanh(np.dot(W_res,r3[:, i])+np.dot(W_in, u_predict[:, i]))
            r4[:] = r3[:, i+1]
            r4[1::2] = r3[1::2, i+1]**2
            if i >= 100:
                u_predict[:dim, i+1] = np.dot(W_out, r4)

        rmse_total = u_train[:dim, :Np]-u_predict[:dim, :Np]
        rmse_mean[j] = np.sqrt(np.mean(rmse_total**2))
        plt.figure(j)
        plt.plot(u_train[0,:])
        plt.plot(u_predict[0,:])
    
    return np.mean(rmse_mean)

#-------------------------------------------------------------------------------------------
def prediction(n,par,x_initial,y_initial,W_out,W_res,W_in):    #prediction for new parameter   bifurcation diagran
    
    pred_time=20000
    u_train = np.zeros((4,pred_time))
    u_train = x_initial*np.ones((1, pred_time))
    u_train = np.vstack((u_train, y_initial*np.ones((1, pred_time))))
    u_train = np.vstack((u_train, 0*np.ones((1, pred_time))))
    u_train = np.vstack((u_train, 0.1*np.ones((1, pred_time))))

    u_predict = np.vstack((u_train, par*np.ones((1, pred_time))))

    r3 = np.zeros((n, pred_time))
    r4 = np.zeros((n))
    
    for i in range(pred_time-1):
        r3[:, i+1] = (1-alpha)*r3[:, i]+alpha*np.tanh(np.dot(W_res,r3[:, i])+np.dot(W_in, u_predict[:, i]))
        r4[:] = r3[:, i+1]
        r4[1::2] = r3[1::2, i+1]**2
        
        if i >= wa:
            u_predict[:dim, i+1] = np.dot(W_out, r4)

    pred_data = u_predict[:, wa:]

    return pred_data

#----------------------------------------------------------------------------------------------------
# original system data genertation

train_par1=np.array([0.43,0.42,0.41])

parameter2=0.15  #model_beta
dim=4
x_in=4
y_in=0
x_data=np.transpose(solve_model(train_par1,parameter2,x_in,y_in))
#--------------------------------------------------------------------------------------------------

# ####ESN parameter  making ESN
beta = 5e-8  
k =20 
W_in_a = 0.06 
alpha = 0.3  
eig_rho = 0.5057


Nt = 60000  # training length
wa = 500  # warmup length
Np=400    #prediction length for calculating error
transit = 50  # abondon reservoir length
n = 1200  # reservoir size

ESN_par=[n,alpha,beta,Nt,Np,transit,dim]

W_input=W_inn(n,dim,W_in_a)

W_reservoir= W_ress(k,n,eig_rho)

W_out=create_ESN(ESN_par,x_data,train_par1,W_input,W_reservoir)

error=error_cal(train_par1,x_data,W_out,W_reservoir,W_input) 
print(error)

# #--------prediction for a new parameter--------------------------------------------------------------------

x_inittt = [4,10]   #np.arange(0,15,1)   
y_inittt = [0]  #np.arange(-35,10,2)   

par = [0.4]

for i in range(len(x_inittt)):
    for j in range(len(y_inittt)):

        dataa = prediction(n, par, x_inittt[i], y_inittt[j], W_out, W_reservoir, W_input)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    
    plt.rcParams['xtick.major.width'] = 3
    plt.rcParams['ytick.major.width'] = 3
    plt.rcParams['axes.linewidth'] = 3

    ax.plot(np.arange(len(dataa[0, :])),
        dataa[0, :], linewidth=2, color='midnightblue')
    
    ax.set_xlabel(r'$\rm{time}$')
    ax.set_ylabel(r'$\rm{x_{max}}$')
    plt.text(5000,3,r'$\rm{Initial~condition}= (%s,%s,0,0.1)$'%(x_inittt[i],y_inittt[j]))