# MCMC 
This repo sets up a simple MCMC implemented following the Metropolis algorithm. 

By no means is this production code. We have the following setup
  * The distribution to explore is a highly correlated 2D Gaussian. _the plots display a contour in black_
  * The proposal distribution is a uncorrelated (isotropic) Gaussian conditioned on the current state. The standard deviation can be set arbitrarily (named _rho_ in the code, following Bishop 2006, section 11.2). 
  * Under this setup, the proposal distribution is symmetric
  * To assess convergence, we fit a Gaussian to the samples and calculate KL divergence to the true distribution. (Note that in practise, the true distribution cannot be evaluated in this way)

# Result
Here's some examples of the output
![run1](https://github.com/RobRomijnders/MCMC/blob/master/im/MCMC.gif?raw=true)

As always, I am curious to any comments and questions. Reach me at romijndersrob@gmail.com




