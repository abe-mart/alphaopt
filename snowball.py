# -*- coding: utf-8 -*-

import numpy as np # General numerics
from scipy.integrate import odeint # Integration
from scipy.optimize import minimize # Optimization
import matplotlib.pyplot as plt # Plotting

# Define System Parameters
K0 = 85 # Snowball growth factor 1
beta = 0.07 # Snowball growth factor 2
C_d = 0.3 # Drag coefficient
g = 9.8 # Gravity
rho = 350 # Snow density
theta = np.radians(5) # Slope
rho_a = 0.9 # Air density

# Initial Snowball Conditions
m0 = 10 # Initial mass
v0 = 0 # Initial velocity
r0 = (m0/(4/3.0*np.pi*rho))**(1/3.0) # Initial radius
s0 = 0 # Initial position

# Target force
F_d = 25000

# Set up time array to solve for 30 seconds
t = np.linspace(0,30)

# This function defines the dynamics of our snowball, the equations of motion
# and the rate at which it changes size and mass.
def snowball_dynamics(w,t,p):
    
    # unpack state variables
    M,r,s,v = w
    
    # unpack parameters
    K0,C_d,g,rho,theta,rho_a,beta = p
    
    # Make an array of the right hand sides of the four differential equations that make up our system.
    f = [beta * K0 * np.exp(-beta*t),
         (beta * K0 * np.exp(-beta*t))/(4*np.pi*rho*r**2),
         v,
         (-15*rho_a*C_d)/(56*rho)*1/r*v**2 - 23/7*1/r*beta*K0*np.exp(-beta*t)/(4*np.pi*rho*r**2)*v+5/7*g*np.sin(theta)]
    return f

# This is the objective function of our optimization.  The optimizer will attempt
# to minimize the output of this function by changing the initial snowball mass.
def objective(m0):
    
    # Load parameters
    p = [m0,C_d,g,rho,theta,rho_a,beta]
    
    # Get initial radius from initial mass
    r0 = (m0/(4/3.0*np.pi*rho))**(1/3.0)
    
    # Set initial guesses
    w0 = [m0,r0,s0,v0]
    
    # Integrate forward for 60 seconds
    sol = odeint(snowball_dynamics,w0,t,args=(p,))
    
    # Calculate kinetic energy at the end of the run
    ke = 0.5 * sol[:,0][-1] * sol[:,3][-1]**2

    # Calculate force required to stop snowball in one snowball radius
    F = ke / sol[:,1][-1]
    
    # Compare to desired force : This should equal zero when we are done
    obj = (F - F_d)**2
    
    return obj

# Call optimization using the functions defined above
res = minimize(objective,m0,options={'disp':True})    

# Get optimized initial mass from solution
m0_opt = res.x[0]

# Calculate optimized initial radius from initial mass
r0_opt = (m0_opt/(4/3.0*np.pi*rho))**(1/3.0)

print('Initial Mass: ' + str(m0_opt) + ' kg (' + str(m0_opt*2.02) + ' lbs)')
print('Initial Radius: ' + str(r0_opt*100) + ' cm (' + str(r0_opt*39.37) + ' inches)')

# Just to prove to ourselves that the answer is correct, let's calculate
# the final force using the optimized initial conditions

# Set initial conditions
w0 = [m0_opt,r0_opt,s0,v0]

# Load parameters
p = [m0_opt,C_d,g,rho,theta,rho_a,beta]

# Integrate forward
sol = odeint(snowball_dynamics,w0,t,args=(p,))

# Get kinetic energy
ke = 0.5 * sol[:,0][-1] * sol[:,3][-1]**2

# Get final stopping force
F = ke / sol[:,1][-1]
print('Final Force: ' + str(F))

# Final Position
print('Final Position: ' + str(sol[:,2][-1]))
print('Final Velocity: ' + str(sol[:,3][-1]))

# And some plots of the results
plt.figure()
plt.plot(t,sol[:,0],label='Mass')
plt.plot(t,sol[:,1],label='Radius')
plt.plot(t,sol[:,2],label='Position')
plt.plot(t,sol[:,3],label='Velocity')
plt.title('Snowball')
plt.xlabel('Time (s)')
plt.legend()