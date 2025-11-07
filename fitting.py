#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 07:58:51 2022

@author: weijunyuan
"""

import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import curve_fit


x_data = np.array([1,2,3,4,5,6,7,8,9,10])
y_data = np.array([0.09,0.33,0.77,1.5,2.8,22,66,288,858,19547])



def exp_i(x,A,B,C):
    return A*np.exp(B*x)+C
    #return A*x**B+C

def linear(x,m,b):
    return m*x+b
"""
popt, pcov = curve_fit(exp_i,x_data,y_data)
x = np.arange(1,10.5,0.2)
plt.scatter(x_data, y_data/60)
plt.xlabel("atom number")
plt.ylabel("computational time (mins)")
plt.plot(x,exp_i(x,popt[0],popt[1],popt[2])/60)
plt.savefig("/Users/weijunyuan/OneDrive - HKUST Connect/paper_book_pdf/courses/quantum_optics/computation_time.png", dpi = 300,transparent = True)
plt.show()
"""
x_data = np.log(np.arange(1,6))
y_data = np.array([0.9975041614583402, 1.9999668333333442, 3.2248249098353554, 4.857354446893414, 6.878656796220639])

#x_data = np.log(np.arange(1,9))

#y_data = np.log(np.array([1.0, 1.0, 1.0731646300205278, 1.2103230256940611, 1.3689026370094546, 1.5372614794808088, 1.7108713372495945, 1.8875523443251083]))
popt, pcov = curve_fit(linear,x_data,y_data)
x = np.arange(1,4,0.2)
plt.scatter(x_data, y_data)
plt.xlabel("Atom number N")
plt.ylabel(r"$R_{max}/R(t=0)$")
plt.plot(x,linear(x,popt[0],popt[1]),'g-')
print(popt[0],popt[1])
#plt.savefig("/Users/weijunyuan/OneDrive - HKUST Connect/paper_book_pdf/courses/quantum_optics/R_max_scale_N.png", dpi = 300,transparent = True)
plt.show()

"""
def exp_d(x,A,B,C):
    return A*np.exp(-B*x)+C
x_data = np.array([0,0.2,0.4,0.6,0.8,1])
y_data = np.array([1.3689026370094546, 1.192817210064132, 1.0922840868626325, 1.0358635633215645, 1.0078233930299771, 1.0])
popt, pcov = curve_fit(exp_d,x_data,y_data)
plt.scatter(x_data, y_data)
x = np.arange(0,1.3,0.1)
plt.xlabel(r'$\Gamma_0/\Gamma_c$')
plt.ylabel(r"$R_{max}/R(t=0)$")
plt.plot(x,exp_d(x,popt[0],popt[1],popt[2]),'g-')
print(popt[0],popt[1])
#plt.savefig("/Users/weijunyuan/OneDrive - HKUST Connect/paper_book_pdf/courses/quantum_optics/R_max_scale_gamma.png", dpi = 300,transparent = True)
plt.show()
"""
"""
N_data = np.arange(1,9)
t_data = np.array([0.4204280376434326,1.5902106761932373,3.771937847137451,7.4677910804748535,14.248285293579102,118.04395127296448,344.35187816619873,1288.877180814743])
t_2_data = np.array([3.733560085296631,4.990548849105835,6.610157012939453,8.454587936401367,11.307631969451904,22.85589098930359,417.7491250038147,600.497663974762])
plt.scatter(N_data,t_data/60,label = "solving master eq.")
plt.scatter(N_data,t_2_data/60,alpha = 0.8,label = "quantum jump method")
plt.xlabel("atom number N")
plt.ylabel("computation time (mins)")
plt.legend()
plt.savefig("/Users/weijunyuan/OneDrive - HKUST Connect/paper_book_pdf/courses/quantum_optics/computation_time_1.png", dpi = 300,transparent = True)
"""