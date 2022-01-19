#! /usr/bin/env python

##########################################################################################
# jumping_ODE.py
#
# Simulation of the pogo-stick style jumping robot
#
#
# Created: 03/02/21
#   - Joshua Vaughan
#   - joshua.vaughan@louisiana.edu
#   - https://userweb.ucs.louisiana.edu/~jev9637/
#
# Modified:
#   * 
#
##########################################################################################

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class PogoODE():
    def __init__(self, ma, mp, k, z, A_max, V_max, Distance, Spacing):
        self.g = 0.276 * 9.81           # m/s**2
        self.ma = ma            # actuator mass (kg) 
        self.mp = mp            # rod mass (kg) 
        self.m = ma + mp        # total mass (kg)
        
        self.k = k              # Spring consant (N/m)
        self.wn = np.sqrt(self.k / self.m) # Natural Freq (rad)
        
        # Calculate damping coeff
        self.z = z              # damping ratio (0.01 from paper)
        self.c = 2 * self.z * self.wn * self.m

        # Set input paramters
        self.A_max = A_max       # 46-92 N peak force on data sheet
        self.V_max = V_max       # Maximum velocity 
        self.Distance = Distance # Move distance of actuator (m)
        self.Spacing = Spacing   # Time spacing between two bang-bang inputs (s)
        
        # These are the times for a bang-coast-bang input 
        t1 = 0
        t2 = (self.V_max / self.A_max) + t1
        t3 = (self.Distance / self.V_max) + t1
        t4 = (t2 + t3) - t1
        self.end_time = t4

        if t3 <= t2: # command should be bang-bang, not bang-coast-bang
            t2 = np.sqrt(self.Distance / self.A_max) + t1
            t3 = 2 * np.sqrt(self.Distance / self.A_max) + t1
            self.end_time = t3


    def eq_of_motion(self, t, w):
        """
        Defines the differential equations for the coupled spring-mass system.

        Arguments:
            w :  vector of the state variables:
              w = [x, x_dot, x_act, x_act_dot]
            t :  time
            p :  vector of the parameters:
              p = [mp, ma, k, c, g]
        """
        x, x_dot, x_act, x_act_dot = w
        
        # determine contact with ground or not
        if x > 0: 
            contact = 0
        else:
            contact = 1
            
        # Create f = (x1',x2',x3',x4'):
        sys_ODE = [x_dot,
                   -contact / self.m * (self.k * x + self.c * x_dot) - self.ma / self.m * self.jump_act(t) - self.g,
                   x_act_dot,
                   self.jump_act(t)]
        
        return sys_ODE


    def jump_act(self, t):
        '''
        Jumping actuator input
        ''' 
                    
        # Bang-Bang-style jumping
        jump_neg = self.accel_input(t, self.A_max, self.V_max, self.Distance)
        jump_pos = self.accel_input(t, self.A_max, self.V_max, (0.008), Start_time=self.end_time + self.Spacing)
        
        jump = (-jump_neg + jump_pos)
        
        return jump


    def accel_input(self, t, Amax, Vmax, Distance, Start_time=0):
        '''
        # Function returns acceleration at a given timestep based on user input
        #
        # Amax = maximum accel, assumed to besymmetric +/-
        # Vmax = maximum velocity, assumed to be symmetric in +/-
        # Distance = desired travel distance 
        # StartTime = Time command should begin
        # CurrTime = current time 
        # Shaper = array of the form [Ti Ai] - matches output format of shaper functions
        #           in toolbox
        #          * If no Shaper input is given, then unshaped in run
        #          * If Shaper is empty, then unshaped is run
        #
        #
        # Assumptions:
        #   * +/- maximums are of same amplitude
        #   * command will begin at StartTime (default = 0)
        #   * rest-to-rest bang-coast-bang move (before shaping)
        #
        '''

        # These are the times for a bang-coast-bang input 
        t1 = Start_time
        t2 = (Vmax / Amax) + t1
        t3 = (Distance / Vmax) + t1
        t4 = (t2 + t3) - t1
        end_time = t4

        if t3 <= t2: # command should be bang-bang, not bang-coast-bang
            t2 = np.sqrt(Distance / Amax) + t1
            t3 = 2 * np.sqrt(Distance / Amax) + t1
            end_time = t3
        
            accel = Amax * (t > t1) - 2 * Amax * (t > t2) + Amax * (t > t3)

        else: # command is bang-coast-bang
            accel = Amax * (t > t1) - Amax * (t > t2) - Amax * (t > t3) + Amax * (t > t4)

        return accel


    
    def run_simulation(self, x0, duration=1.0, max_step=0.001):
        """
        Run the simluation 
        
        Arguments
          x0 : array of initial conditions 
               [x_init, x_dot_init, x_act_init, x_act_dot_init]
          duration : how long to run for each simulation
          
        Returns
          solution : the solution of the ODE
        """
        
        # ODE solver parameters
        abserr = 1.0e-6
        relerr = 1.0e-6
        numpoints = int(duration / max_step + 1)
        
        # Create the time samples for the output of the ODE solver.
        t = np.linspace(0, duration, numpoints)
        
        # Call the ODE solver -- This is the actual simulation part
        solution = solve_ivp(fun=self.eq_of_motion, 
                            t_span=[0, duration], 
                            y0=x0, 
                            dense_output=True,
                            t_eval=t, 
                            max_step=max_step, 
                            atol=abserr, 
                            rtol=relerr,
                            # method='Radau',
                            )

        if not solution.success: 
            # The ODE solver failed. Notify the user and print the error message
            print('ODE solution terminated before desired final time.')
            print('Be *very* careful trusting the results.')
            print('Message: {}'.format(solution.message))

        # Parse the time and response arrays from the OdeResult object
        sim_time = solution.t
        response = solution.y
        return sim_time, response


if __name__ == "__main__":

    m_rod = 0.175               # mass of the pogo-stick rod (kg)
    m_act = 1.003               # mass of the pogo-stick rod (kg)
    mass = m_rod + m_act        # total mass (kg)
    f = 11.13                   # natural freq. (rad)
    wn = f * (2 * np.pi)        # Robot frequency (rad/s)
    zeta = 0.01                 # Robot damping ratio
    c = 2 * zeta * wn * mass    # Calculate damping coeff
    k = mass * wn**2            # Calulate spring constant
    # k = 150

    A_max = 63.2      # max acceleration of actuator (m/s^2)
    V_max = 2.0       # max velocity of actuator (m/s)
    Distance = 0.008  # Distance to move actuator in jump command (m)
    Spacing = 0.75 * (1 / f)  # Space commands by 0.5*period of oscillation
    
    pogo_stick = PogoODE(m_act, m_rod, k, zeta, A_max, V_max, Distance, Spacing)
    
    x_init = 0.0
    x_dot_init = 0.0
    x_act_init = 0.0
    x_act_dot_init = 0.0
    
    x0 = [x_init, x_dot_init, x_act_init, x_act_dot_init]

    sim_time, timeseries = pogo_stick.run_simulation(x0, duration=1.0)

    
    ##################################################################################################
    # Set the plot size - 3x2 aspect ratio is best
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca()

    # Define the X and Y axis labels
    plt.xlabel(r'Time (s)', fontsize=22, weight='bold', labelpad=5)
    plt.ylabel(r'Height (m)', fontsize=22, weight='bold', labelpad=10)

    plt.plot(sim_time, timeseries[0], linewidth=2, linestyle='-', label=r"Rod Position")

    # uncomment below and set limits if needed
    # plt.xlim(0,4)
    # plt.ylim(bottom=None, top=1.5)

    # Create the legend, then fix the fontsize
    leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
    ltext  = leg.get_texts()
    plt.setp(ltext,fontsize=18)

    # Adjust the page layout filling the page using the new tight_layout command
    plt.tight_layout(pad=0.5)

    ##################################################################################################
    # Set the plot size - 3x2 aspect ratio is best
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca()

    # Define the X and Y axis labels
    plt.xlabel(r'Time (s)', fontsize=22, weight='bold', labelpad=5)
    plt.ylabel(r'Height (m)', fontsize=22, weight='bold', labelpad=10)

    plt.plot(sim_time, timeseries[1], linewidth=2, linestyle='-', label=r"Rod Velocity")

    # uncomment below and set limits if needed
    # plt.xlim(0,4)
    # plt.ylim(bottom=None, top=1.5)

    # Create the legend, then fix the fontsize
    leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
    ltext  = leg.get_texts()
    plt.setp(ltext,fontsize=18)

    # Adjust the page layout filling the page using the new tight_layout command
    plt.tight_layout(pad=0.5)

    ##################################################################################################
    # Set the plot size - 3x2 aspect ratio is best
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca()

    # Define the X and Y axis labels
    plt.xlabel(r'Time (s)', fontsize=22, weight='bold', labelpad=5)
    plt.ylabel(r'Height (m)', fontsize=22, weight='bold', labelpad=10)

    plt.plot(sim_time, timeseries[2], linewidth=2, linestyle='-', label=r"Act Position")

    # uncomment below and set limits if needed
    # plt.xlim(0,4)
    # plt.ylim(bottom=None, top=1.5)

    # Create the legend, then fix the fontsize
    leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
    ltext  = leg.get_texts()
    plt.setp(ltext,fontsize=18)

    # Adjust the page layout filling the page using the new tight_layout command
    plt.tight_layout(pad=0.5)

    ##################################################################################################
    # Set the plot size - 3x2 aspect ratio is best
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca()

    # Define the X and Y axis labels
    plt.xlabel(r'Time (s)', fontsize=22, weight='bold', labelpad=5)
    plt.ylabel(r'Height (m)', fontsize=22, weight='bold', labelpad=10)

    plt.plot(sim_time, timeseries[3], linewidth=2, linestyle='-', label=r"Act Velocity")

    # uncomment below and set limits if needed
    # plt.xlim(0,4)
    # plt.ylim(bottom=None, top=1.5)

    # Create the legend, then fix the fontsize
    leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
    ltext  = leg.get_texts()
    plt.setp(ltext,fontsize=18)

    # Adjust the page layout filling the page using the new tight_layout command
    plt.tight_layout(pad=0.5)

    plt.show()
