#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 22:19:57 2022

@author: weijunyuan
"""


## The program is for quantum optics class

import matplotlib.pyplot as plt
import numpy as np
from numpy import array,matmul,matrix,dot,transpose,identity,average,kron,zeros,trace
import time
import random
from scipy.signal import lfilter
alpha = 5  # gamma_c/gamma0
def f(x,t):
    return -(1+alpha)*x

def anal_result(t):
    return np.exp(-t*(1+alpha))


sigma_p = array([[0,1],[0,0]])
sigma_m = array([[0,0],[1,0]])
sigma_ee = array([[1,0],[0,0]])

def f_3 (rho,t=0):
    new_mat = 2*matmul(sigma_m,matmul(rho,sigma_p))-matmul(sigma_p,matmul(sigma_m,rho))-matmul(rho,matmul(sigma_p,sigma_m))
    fx = new_mat*(1+alpha)/2
    return fx
# rho = array([[0,0],[1,1]])
# print(f_3(rho))

def q1_matrix():
    #main function of question 1
    a = 0     # starting time
    b = 1.5    # end time
    N = 1000  # time step
    h = (b-a)/N  
    t_list = np.linspace(a,b,N)

    rhoee_list = []
    rhoeg_list = []
    rhoge_list = []
    rhogg_list = []
    tem_r = np.array([[1.0,0.0],[0.0,0.0]],dtype = float) # initial condition
    for nt in t_list:
        #multivariable Runge-Kutta method
        ee = tem_r[0,0]
        eg = tem_r[0,1]
        ge = tem_r[1,0]
        gg = tem_r[1,1]
        rhoee_list.append(ee)
        rhoeg_list.append(eg)
        rhoge_list.append(ge)
        rhogg_list.append(gg)
        
        k1 = h*f_3(tem_r,nt)
        k2 = h*f_3(tem_r+0.5*k1,nt+0.5*h)    
        k3 = h*f_3(tem_r+0.5*k2,nt+0.5*h)
        k4 = h*f_3(tem_r+k3,nt+h)
        tem_r +=(k1+2*k2+2*k3+k4)/6
    plt.plot(t_list,rhoee_list,"r-",label = "numerical")
    #plt.plot(t_list,rhogg_list,"b-",label = "rho_gg")
    # plt.plot(t_list,rhoge_list,"g-",label = "rho_ge")
    # plt.plot(t_list,rhoge_list,"y-",label = "rho_eg")
    plt.legend()
    plt.ylabel(r'$\rho_{ee}$')
    plt.xlabel(r"$\Gamma_{0}$t")
    analytical = anal_result(t_list)
    plt.plot(t_list,analytical,label = "analytical ")
    plt.legend()
    plt.savefig("/Users/weijunyuan/OneDrive - HKUST Connect/paper_book_pdf/courses/quantum_optics/numerical_plot_oneatom.png", dpi = 300)
    
#q1_matrix()


############################################
#Quantum jump for single atom
def delta_p(phi,c_m,c_m_d,h):
    
    return dot(phi.conj(),matmul(c_m_d,matmul(c_m,phi)))*h.real
def phi_1(phi_t,action):
    return matmul(action,phi_t)
        
def quantum_jump():
    a = 0
    b = 1
    N = 1000
    h = (b-a)/N     ### time interval
    c_m = sigma_m*np.sqrt((1+alpha))
    c_m_d = sigma_p*np.sqrt(1+alpha)
    H_eff = -(1j/2)*matmul(c_m_d,c_m)
    action = identity(2)- 1j*H_eff*h
    r_1 = 0
    phi_0 = array([1,0])
    phi_e = []
    phi_g = []
    phi = phi_0
    rho_ee = [dot(phi_0[0],phi_0[0])]
    rho_gg = [dot(phi_0[1],phi_0[1])]
    for i in range(1,N):
        #r_1 = random.uniform(0,1)
        r_1 = np.random.rand()
        d_p = delta_p(phi,c_m,c_m_d,h)
        if r_1 > d_p:
            new_phi = phi_1(phi,action)/np.sqrt(1-d_p)
        else:
            new_phi = matmul(c_m,phi)/np.sqrt(d_p/h)
        #print(new_phi)
        phi_e.append(new_phi[0])
        phi_g.append(new_phi[1])
        rho_ee.append(dot(new_phi[0].conj(),new_phi[0]).real)
        phi = new_phi
    return rho_ee
#quantum_jump()



def single_atom_qj_solver():
    a = 0
    b = 1
    N = 1000
    t_list = np.linspace(a,b,N)
    rho_ee_list = []
    for i in range(50):
        rho_ee = quantum_jump()
        plt.plot(t_list,rho_ee)
        rho_ee_list.append(rho_ee)
    rho_ee_list = array(rho_ee_list)
    t_list = np.linspace(a,b,N)
    rho_ee_avg = average(rho_ee_list,axis = 0)
    #plt.plot(t_list,rho_ee_avg,"r-",label = "1 quantum trajectories")
    analytical = anal_result(t_list)
    #plt.plot(t_list,analytical,'b-',label = "analytical result")
    plt.legend()
    plt.ylabel(r'$\rho_{ee}$')
    plt.xlabel(r"$\Gamma_{0}$t")
    #plt.savefig("/Users/weijunyuan/OneDrive - HKUST Connect/paper_book_pdf/courses/quantum_optics/quantum_jump_single_atom_1.png", dpi = 300)
    #plt.savefig("/Users/weijunyuan/OneDrive - HKUST Connect/paper_book_pdf/courses/quantum_optics/quantumtrajecties.png", dpi = 300)
single_atom_qj_solver()

################ multiatoms

def initial_state(N):
    phi_0 = array([1.,0.],dtype = "complex128")
    phi_N = phi_0
    for i in range(N):
        phi_N = kron(phi_N,phi_0)
    return phi_N


def I_N(N):
    return identity(2**N)

def Sigma_m(N,j):
    return kron(I_N(j-1),kron(sigma_m,I_N(N-j)))

def Sigma_ee(N,j):
    return kron(I_N(j-1),kron(sigma_ee,I_N(N-j)))
    
def Sigma_p(N,j):
    return kron(I_N(j-1),kron(sigma_p,I_N(N-j)))
def rho0_N(rho_0,N):
    rho_N = identity(1)
    for i in range(N):
        rho_N = kron(rho_N,rho_0)
    return rho_N
gamma_c = 1
gamma_0 = 0
def H_N(rho,N,Gamma_0,t=0):
    H_jj = zeros((2**N,2**N))
    H_ij = zeros((2**N,2**N))
    Rho = rho ### initial condition
    for i in range(1,N+1):
        for j in range(1,N+1):
            if i == j:
                H_jj += 2*matmul(Sigma_m(N, j),matmul(Rho,Sigma_p(N, j)))- \
                        matmul(Sigma_p(N, j),matmul(Sigma_m(N, j),Rho))- \
                        matmul(Rho,matmul(Sigma_p(N, j),Sigma_m(N, j)))
            else:
                H_ij += 2*matmul(Sigma_m(N, j),matmul(Rho,Sigma_p(N, i)))- \
                        matmul(Sigma_p(N, i),matmul(Sigma_m(N, j),Rho))- \
                        matmul(Rho,matmul(Sigma_p(N, i),Sigma_m(N, j)))
    H = H_jj*(gamma_c+Gamma_0)+H_ij*gamma_c
    return H/2
# rhoi = rho0_N(rho_0, 2)
# print(H_N(rhoi,2))
def pop_sum(N):
    ### give the population operator for N atoms
    pop_operator = zeros((2**N,2**N))
    for i in range(1,N+1):
        pop_operator += Sigma_ee(N, i)
    return pop_operator
def exp_value(rho,operator,N):
    ### find the expectation value of an operator for given rho
    expect = trace(matmul(rho,operator))
    return expect

rho_0 = array([[1,0],[0,0]])
def multiatom_solver(N=1,Gamma_0=0):
    a = 0     # starting time
    b = 1.5  # end time
    n = 300  # time step
    h = (b-a)/n  
    t_list = np.linspace(a,b,n)
    Rho_0 = rho0_N(rho_0, N)
    population = pop_sum(N)
    tem_r = Rho_0# initial condition
    pop_list = []
    rate_list = []
    for nt in t_list:
        #multivariable Runge-Kutta method
        pop_exp = exp_value(tem_r, population, N)
        pop_list.append(pop_exp)
        k1 = h*H_N(tem_r,N,Gamma_0,nt)
        k2 = h*H_N(tem_r+0.5*k1,N,Gamma_0,nt+0.5*h)    
        k3 = h*H_N(tem_r+0.5*k2,N,Gamma_0,nt+0.5*h)
        k4 = h*H_N(tem_r+k3,N,Gamma_0,nt+h)
        tem_r +=(k1+2*k2+2*k3+k4)/6
    #plt.plot(t_list,pop_list,"r-",label = "N = " + str(N)+" Master Eq. ")
    for i in range(1,n):
        rate = (pop_list[i]-pop_list[i-1])/h
        rate_list.append(rate)
    rate_list = -1*array(rate_list)

    return pop_list,rate_list


"""
a = 0     # starting time
b = 1.5  # end time
n = 500  # time step
h = (b-a)/n  
t_list = np.linspace(a,b,n)
start = time.time()
N = 9
exp_ee,rate = multiatom_solver(N,gamma_0)
plt.plot(t_list,exp_ee,"r-",label = "N = " + str(N)+" Master Eq. ")
end = time.time()
print(end-start)
"""
"""
a = 0     # starting time
b = 1.5  # end time
n = 500  # time step
h = (b-a)/n  
t_list = np.linspace(a,b,n)
for i in range(1,9):
    exp_ee,rate = multiatom_solver(i,gamma_0)
    plt.plot(t_list,exp_ee,label = "N = "+str(i))
    plt.legend()
    plt.xlabel(r'$\Gamma_c t$')
    plt.ylabel(r'$<\sum_{i} \sigma_{ee}^i>$')
plt.savefig("/Users/weijunyuan/OneDrive - HKUST Connect/paper_book_pdf/courses/quantum_optics/multiatom_population.png", dpi = 300)
plt.show()
"""
def rate_N(N):
    a = 0     # starting time
    b = 1  # end time
    n = 300  # time step
    t_list = np.linspace(a,b,n)
    rate_ll = []
    pop_ll = []
    for i in range(1,N):
        pop_i,rate_i = multiatom_solver(i)
        plt.plot(t_list[0:n-1],rate_i,label = "N = "+str(i))
        rate_ll.append(rate_i)
        pop_ll.append(pop_i)
    plt.ylabel(r"R")
    plt.xlabel(r"$\Gamma_c$ t")
    plt.legend()
    #plt.savefig('/Users/weijunyuan/OneDrive - HKUST Connect/paper_book_pdf/courses/quantum_optics/super_radiant_burst.png',dpi = 300,transparent = True)
    plt.show()
    rate_max = []
    for rate in rate_ll:
        rate_max_i = max(rate)
        rate_max.append(rate_max_i)
    n = np.arange(1,N)
    plt.scatter(n,rate_max)
    plt.xlabel("N")
    plt.ylabel(r"$R_{max}$")
    #plt.savefig('/Users/weijunyuan/OneDrive - HKUST Connect/paper_book_pdf/courses/quantum_optics/super_radiant_scale_N.png',dpi = 300,transparent = True)
    print(rate_max)

rate_N(N = 6)

def rate_gamma():
    a = 0     # starting time
    b = 1  # end time
    n = 300  # time step
    t_list = np.linspace(a,b,n)
    rate_ll = []
    pop_ll = []
    Gamma = [0,0.2,0.4,0.6,0.8,1]
    for gamma_0 in Gamma:
        pop_i,rate_i = multiatom_solver(N = 5,Gamma_0 = gamma_0)
        plt.plot(t_list[0:n-1],rate_i,label = r"$\Gamma_{0}/\Gamma_{c}$ = "+str(gamma_0))
        rate_ll.append(rate_i)
        pop_ll.append(pop_i)
    plt.ylabel(r"R")
    plt.xlabel(r"$\Gamma_c$ t")
    plt.legend()
    #plt.savefig('/Users/weijunyuan/OneDrive - HKUST Connect/paper_book_pdf/courses/quantum_optics/super_radiant_burst_gamma_0.png',dpi = 300,transparent = True)
    plt.show()
    rate_max = []
    for rate in rate_ll:
        rate_max_i = max(rate)
        rate_max.append(rate_max_i)
    plt.scatter(Gamma,rate_max)
    plt.xlabel(r"$\Gamma_0/\Gamma_c$")
    plt.ylabel(r"$R_{max}$")
    #plt.savefig('/Users/weijunyuan/OneDrive - HKUST Connect/paper_book_pdf/courses/quantum_optics/super_radiant_scale_gamma_0.png',dpi = 300,transparent = True)
    print(rate_max)
#rate_gamma()

########===========multiatom quantum jump=========##############

def gamma_matrix(N,Gamma_0,Gamma_c):
    ###define the gamma matrix 
    Gamma = identity(N)*(Gamma_0)+np.ones((N,N))*Gamma_c
    return Gamma

def collective_jump_O(N,w,v):
    O_list = []
    O_d_list = []
    sigma_p_list = []
    sigma_m_list = []
    for i in range(1,N+1):
        sigma_p_list.append(Sigma_p(N,i))
        sigma_m_list.append(Sigma_m(N,i))
    for l in range(N):
        O_d_l = zeros((2**N,2**N))
        for i in range(0,N):
            O_d_l = O_d_l + v[:,l][i]*sigma_p_list[i]
        O_d_list.append(O_d_l*np.sqrt(w[l]))
        #O_list.append(np.sqrt(w[l])*np.transpose(np.conjugate(O_d_l)))
    for l in range(N):
        O_l = zeros((2**N,2**N))
        for i in range(0,N):
            O_l = O_l + np.conjugate(v[:,l][i])*sigma_m_list[i]
        O_list.append(O_l*np.sqrt(w[l]))
    return O_list,O_d_list
def delta_p_N(N,phi,c_m,c_m_d,h):
    d_p_list = []
    d_p = 0
    for i in range(N):
        d_p_m = dot(phi.conj(),matmul(c_m_d[i],matmul(c_m[i],phi)))*h
        d_p_list.append(d_p_m)
        d_p = d_p+d_p_m
    return d_p,d_p_list

def H_eff_gen(N,c_m,c_m_d):
    H = zeros((2**N,2**N))
    for i in range(N):
        H = H+ matmul(c_m_d[i],c_m[i])
    return -1j/2*H
def phi_0_N(N):
    ### return the initial state fir 
    phi_list = [1]
    for i in range(2**N-1):
        phi_list.append(0)
    return np.array(phi_list)
def interval_gen(N,d_p,d_p_list):
    d_p_array = np.array(d_p_list)
    normalized_array = d_p_array/d_p
    interval_b = 0  ### interval upper boundary
    b_list = []   ### boundary list
    for d_p_m in normalized_array:
        interval_b = interval_b+d_p_m
        b_list.append(interval_b.real)
    return b_list
def pop_2(phi,population):
    return dot(phi.conj(),matmul(population,phi))
def quantum_jump_N(N,N_i,h,c_m,c_m_d,action,phi_0,population):
    r_1 = 0
    r_2 = 0
    phi = phi_0
    d_p,d_p_list= delta_p_N(N,phi,c_m,c_m_d,h)
    pop_list = []
    for i in range(N_i):
        r_1 = random.uniform(0,1)
        d_p,d_p_list= delta_p_N(N,phi,c_m,c_m_d,h)
        if r_1 > d_p:
            new_phi = phi_1(phi,action)/np.sqrt(1-d_p)
        else:
            r_2 = random.uniform(0,1)
            interval_list= interval_gen(N,d_p,d_p_list)
            j = 0
            true = True
            while true:
                interval = interval_list[j]
                if r_2 <= interval :
                    new_phi = phi_1(phi,c_m[j])/np.sqrt(d_p_list[j]/h)
                    true = False
                else:
                    j = j+1
        # phi_matrix = np.mat(phi)
        # tem_r = matmul(np.transpose(np.conjugate(phi_matrix)),phi_matrix)   ### density matrix
        #pop_exp = exp_value(tem_r, population, N)      
        pop_exp = pop_2(phi,population)
        phi = new_phi
        pop_list.append(pop_exp.real)
    return pop_list


def multiatom_qj_solver(N = 1,n = 100):
    a = 0
    b = 1.5
    N_i = 200
    h = (b-a)/N_i
    t_list = np.linspace(a,b,N_i)
    rho_ee_list = []
    Gamma_N = gamma_matrix(N,gamma_0,gamma_c)
    w,v = np.linalg.eig(Gamma_N)
    for i in range(N):
        if w[i]< 0:
            w[i] = 0
    c_m,c_m_d = collective_jump_O(N,w,v)
    H_eff = H_eff_gen(N, c_m, c_m_d)
    action = identity(2**N)- 1j*H_eff*h
    population = pop_sum(N)
    phi_0 = phi_0_N(N)  ### the initial state is all atoms get excited
    for i in range(n):
        rho_ee = quantum_jump_N(N,N_i,h,c_m,c_m_d,action,phi_0,population)
        print("done "+str(i))
        rho_ee_list.append(rho_ee)
    rho_ee_list = array(rho_ee_list)
    t_list = np.linspace(a,b,N_i)
    rho_ee_avg = average(rho_ee_list,axis = 0)
    plt.plot(t_list,rho_ee_avg,"b-",label = str(n)+" quantum trajectories")
    #analytical = anal_result(t_list)
    #plt.plot(t_list,analytical,'b-',label = "analytical result")
    plt.legend()
    plt.ylabel(r'$<\sum_{i} \sigma_{ee}^i>$')
    plt.xlabel(r"$\Gamma_{c}$t")
    # rate_list = []
    # for i in range(1,N_i):
    #     rate = (rho_ee_avg[i]-rho_ee_avg[i-1])/h
    #     rate_list.append(rate)
    # rate_list = -1*array(rate_list)
    # n = 15  # the larger n is, the smoother curve will be
    # b = [1.0 / n] * n
    # a = 1
    # y = lfilter(b,a,rate_list)
    # plt.plot(t_list[0:N_i-1],y
    plt.savefig("/Users/weijunyuan/OneDrive - HKUST Connect/paper_book_pdf/courses/quantum_optics/quantum_jump_"+str(N)+"_atom_"+str(n)+".png", dpi = 300)
    plt.show()

#exp_ee,rate = multiatom_solver(4,gamma_0)
# start = time.time() 
#multiatom_qj_solver(12,n = 50)
# end = time.time()
# print(end-start)




