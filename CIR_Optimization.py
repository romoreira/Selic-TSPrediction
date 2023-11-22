#https://github.com/Sumit112192/Interest_Rate_Models/blob/main/README.md

import numpy as np
import scipy
import matplotlib.pyplot as plt

#The below reads the rates from file fedRates.dat
#Simply after this line, you would be having a 1D list of rates
rates = np.array([float(line.strip().split()[4])/100 for line in open("../fedRates.dat").readlines()])

dt = 1/12 #Since the spot rates are 1 month apart.


Y = []
Z = []
for i in range(len(rates)-1):
    Y.append( (rates[i+1]-rates[i]) / np.sqrt( rates[i] ) ) #Assuming positive interest rates
    Z.append( [dt/np.sqrt(rates[i]), dt*np.sqrt(rates[i])])


beta = scipy.linalg.lstsq(Z, Y)
N = len(rates)
alpha = -beta[0][1]
mu = beta[0][0]/alpha
res = beta[1]
sigma = np.sqrt(res)/np.sqrt(dt*N)
print("OLS Estimations")
print("Alpha = ", alpha, "Mu = ", mu, "sigma = ", sigma)

def CIRlog(guess):
    alpha = guess[0]
    mu = guess[1]
    sigma = guess[2]
    c = (2*alpha)/((sigma**2)*(1-np.exp(-alpha*dt)))
    u = c*np.exp(-alpha*dt)*rates[:-1]
    v = c*rates[1:]
    q = (2*alpha*mu)/(sigma**2)-1
    z = 2*np.sqrt(u*v)
    bf = scipy.special.ive(q, z)

    logL = (N-1)*np.log(c) - np.sum(u+v-0.5*q*np.log(v/u)-np.log(bf)-z)
    return -logL

print("MLE Optimized Parameters")
optimizedParam = scipy.optimize.fmin(CIRlog, (alpha, mu, sigma))
print(optimizedParam)