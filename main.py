# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 13:22:54 2017

@author: rob
"""


from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from util import plot_cov_ellipse
from numpy.linalg import det

"""Define our helper classes"""

class proposal():
  def __init__(self,rho,dim=2):
    self.dim = dim  #dimensionality of the dsitribution to sample from
    self.rho = rho  # scale parameter for the proposal distribution
    self.q = multivariate_normal(mean=[0]*dim,cov=np.square(rho)*np.eye(dim))
  def propose(self,x):
    """x is the current state"""
    assert self.dim in x.shape,'vector x mismatches dimensionality'
    return self.q.rvs()+x

class distro():
  def __init__(self,mean,cov):  
    self.dim = mean.shape[0]
    assert cov.shape[0] == self.dim, 'Mean and covariance mismatch in dimension'
    self.p = multivariate_normal(mean=mean,cov=cov)
    self.normalization_bias = np.random.uniform(10,100)  #add a random normalization bias to the distribution to simulate real scenarios
  def pdf(self,x):
    assert x.shape[0] == self.dim, 'evaluated state mismatches expected dimension'
    p_bias = self.normalization_bias*self.p.pdf(x)
    return p_bias
  def KL(self,states):
    assert isinstance(states,list)
    m2 = np.mean(states,0)
    s2 = np.cov(states,rowvar=False)+1E-9
    m1 = self.p.mean
    s1 = self.p.cov
    try:
      s2inv = np.linalg.inv(s2)
    except:
      print(s2)
    m2_m1 = m2 - m1
    KLD = 0.5*(det(s2)/det(s1) - self.dim + np.sum(np.multiply(s2inv,s1))+m2_m1.dot(s2inv).dot(m2_m1))
    return KLD
 
class acceptance():
  def __init__(self,p_tilde):
    assert isinstance(p_tilde,distro),'p_tilde must be a distribution of type distro'
    self.p_tilde = p_tilde   #the unnormalized density to sample from
  def accept(self,state_current,state_prop):
    acceptance = self.p_tilde.pdf(state_prop)/self.p_tilde.pdf(state_current)
    acceptance = np.min((1,acceptance))  #The probability we accept the proposed state given the current state
    to_accept = np.random.rand()<acceptance  #The boolean if we accept yes or no
    return to_accept
    
  
    
"""Start with MCMC"""
mean = np.array([2.3, 2.4])
cov = np.array([[1.0,1.1],[1.1,2.0]])

rhos = np.array([0.05,0.1,0.5,1.0])
R = len(rhos)

max_iter = 500

p_tilde = distro(mean,cov)

A = acceptance(p_tilde)

f, ax = plt.subplots(2, R)  
for r in range(R):
  plot_cov_ellipse(cov, pos=mean, volume=.5, ax=ax[0,r])
  ax[0,r].set_xlim([-2,7]) 
  ax[0,r].set_ylim([-2,7])
  ax[1,r].set_xlim([0,max_iter])
  ax[1,r].set_ylim([0,100])
  ax[1,r].set_xlabel('steps')
ax[1,0].set_ylabel('KL divergence')


Qs = [proposal(rho) for rho in rhos]
rej = [0]*R

state = [2+np.random.rand(2) for _ in range(R)]  #Random state to start from. This can be any random start function
states = [[s] for s in state]
for i in range(max_iter):
  if i%50 == 0: print(i)
  for r,rho in enumerate(rhos):
    #use states.append(_.copy()) to prevent Python from changing state in subsequent steps
    state_prop = Qs[r].propose(state[r])
    if A.accept(state[r],state_prop):
      states[r].append(state_prop.copy())
      state[r] = state_prop
      rej[r] += 0
    else:
      states[r].append(state[r].copy())
      rej[r] += 1
      
    ax[0,r].plot(state[r][0],state[r][1],'.',linewidth=0) 
    ax[0,r].set_title('rjct %5.2f'%((rej[r]/float(i+1))))
    if i>2:
      ax[1,r].plot(i,p_tilde.KL(states[r]),'.')
  if i%10 == 0:
    pass
    plt.savefig('im/step%05d.png'%i)


#Now go to the directory and run  (after install ImageMagick)
#  convert -delay 10 -loop 0 *.png MCMC.gif