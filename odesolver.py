# -*- coding: utf-8 -*-
"""
Numerische Mathematik 3 WiSe2023, Prof Hausser
modul <odesolver>
Heewhan Soul, Hajin Suh

This module includes some numerical method to solve the differential equations.
Some basic methods are implemented by Professor Hausser but we added here to use in the project properly.

0. class ODEResult: The functions of differential equation solver following will be constructed with identical 
                                structure and attribute. This class defines them.
1. euler_forward : It solves the differential equation with euler forward method.
2-1. euler_verbessert : It solves the differential equation with improved euler method.
2-2. euler_backward : : It solves the differential equation with euler backward method.
3. rungekutta_4 :It solves the diffential equation with classic Runge Kutta method of 4ks.
4. rungekutta_43 :It solves the diffential equation with Runge Kutta method of 4ks with adaptive stepsize.
5. test_skalar : test rungekutta_43 with scalar example
6. test_system : test rungekutta_43 with vector example

"""
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.integrate import solve_ivp



"""    0. class ODEResult   """  
class ODEResult(dict):
    ''' Container object exposing keys as attributes.
    Bunch objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `bunch["value_key"]`, or by an attribute, `bunch.value_key`'''
    
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError
            
    def __setattr__(self, key, value):
        self[key] = value        
            
    
"""    1. euler_forward   """
def euler_forward(f, t_span, y0, t_eval):
    """
    It solves the diffential equation with euler forward method.
    INPUT 
        f : differential equation to be solved
        t_span : interval of t to be dealt
        y0 : initial value of y
        t_eval : ts to be evaluated
    OUTPUT
        ODEResult
            t : array of evaluated ts
            y : array of evaluated vaules y
    """
    t0 = t_span[0]
    tf = t_span[1]
    # array to save all y
    ys = np.zeros( (len(y0), len(t_eval) ), dtype = float )
    ts = t_eval
    ys[:, 0] = y0
    
    for k in range( len(t_eval) - 1 ):
        # Step of t
        h = ts[k+1] - ts[k]
        # calculated y according to the formula of euler forward method
        ys[:, k+1] = ys[:, k] + h*f(ts[k], ys[:, k])
    
    return ODEResult(t=ts, y=ys)


"""    2-1. euler_verbessert   """
def euler_verbessert(f, t_span, y0, t_eval):
    """
    It solves the diffential equation with improved euler method.
    INPUT 
        f : differential equation to be solved
        t_span : interval of t to be dealt
        y0 : initial value of y
        t_eval : ts to be evaluated
    OUTPUT
        ODEResult
            t : array of evaluated ts
            y : array of evaluated vaules y
            auswertung : number of function evaluations
    """
    t0 = t_span[0]
    tf = t_span[1]
    # array to save all y
    ys = np.zeros( (len(y0), len(t_eval) ), dtype = float )
    ts = t_eval
    ys[:, 0] = y0
    
    anzahl_auswertung = 0
    
    for k in range( len(t_eval) - 1 ):
        # Step of t
        h = ts[k+1] - ts[k]
        # calculated y according to the formula of improved euler method
        k1 = f(ts[k], ys[:,k])
        k2 = f(ts[k] + h/2.*k1, ys[:,k] + h/2)
        
        anzahl_auswertung += 2
        
        ys[:, k+1] = ys[:, k] + h*k2
    
    return ODEResult(t=ts, y=ys, auswertung=anzahl_auswertung)

"""    2-2. euler_backward   """
def euler_backward(f, t_span, y0, t_eval, jac):
    """
    implicit solver for first order initial value problem

    Parameters
    ----------
    f : function with signature dy = f(t, y)
        right hand side of ODE 
    jac: jacobian of f with respect to y
    t_span : tupel (t0, tf)
        left and right boundary of interval on which the IVP is solved
    y0 : ndarray of shape (n,)
        initial value y(t_initial)
    t_eval : ndarray of shape (N+1, )
        

    Returns
    -------
    bunch Object ODEResult with members
        t:  ndarray of shape (N+1, ) 
            discrete time instances
        y:  ndarray of shape(N+1, n)
            approximated values of solution at time instances given in t
        its: ndarray of shape shape (N+1, )
            Newton iterations in each time step

    """
    
    tol1 = 1.e-10
    tol2 = 1.e-10
    maxIt = 10
    
    t0 = t_span[0]
    tf = t_span[1]
    ys = np.zeros( (len(y0), len(t_eval) ), dtype = float )
    ts = t_eval
    ys[:, 0] = y0
    
    id = np.eye(len(y0))
    iterations = np.zeros_like(ts, dtype=int)  
    
    for k in range( len(t_eval) - 1 ):    # time loop
        
        # Newton Iteration
        h = ts[k+1] - ts[k]  # Zeitschritt
        y = ys[:, k].copy()  # Startvektor für Newton-Iteration
        t = ts[k+1]          # nächster Zeitpunkt
        it = 0
        err1 = tol1 + 1
        err2 = tol2 + 1
        while( err1 > tol1 and err2 > tol2 and it < maxIt):
            b = ys[:,k] + h*f(ts[k], y) - y
            A = h*jac(t, y) - id
            delta_y = np.linalg.solve(A, -b)
            
            y += delta_y
            it += 1
            err1 = np.linalg.norm(delta_y)
            err2 = np.linalg.norm(b) 
        
        ys[:, k+1] = y 
        iterations[k] = it
     
    return ODEResult(t=ts, y=ys, its = iterations)



"""    3. rungekutta_4    """
def rungekutta_4(f, t_span, y0, t_eval):
    """
    It solves the diffential equation with classic Runge Kutta method of 4ks.
    INPUT
        f : differential equation to be solved
        t_span : interval of t to be dealt
        y0 : initial value of y
        t_eval : ts to be evaluated
    OUTPUT
        ODEResult
            t : array of evaluated ts
            y : array of evaluated vaules y
            auswertung : number of function evaluations
    """
    ts = t_eval
    # array to save all y
    ys = np.zeros( (len(y0), len(t_eval) ), dtype = float )
    ys[:, 0] = y0
    anzahl_auswertung = 0
    
    for k in range(len(ts) - 1):
        # Calculate ki and y according to fomula of Runge Kutta method
        h =  ts[k+1] - ts[k]
        k1 = f(ts[k], ys[:,k])
        k2 = f(ts[k] + h/2, ys[:,k] + h/2 * k1)
        k3 = f(ts[k] + h/2, ys[:,k] + h/2 * k2)
        k4 = f(ts[k] + h,   ys[:,k] + h * k3)
        ys[:,k+1] = ys[:,k] + 1./6*h*(k1 + 2*k2 + 2*k3 + k4)
        
        anzahl_auswertung += 4
        
    return ODEResult(t=ts, y=ys, auswertung=anzahl_auswertung)


"""    4. rungekutta_43    """
def rungekutta_43(f, t_span, y0, tol=1.e-6):
    """
    It solves the diffential equation with Runge Kutta method of 4ks with adaptive stepsize.
    INPUT
        f : differential equation to be solved
        t_span : interval of t to be dealt
        y0 : initial value of y
        tol : Tolerance 
    OUTPUT
        ODEResult
            t : array of evaluated ts
            y : array of evaluated vaules y    
            auswertung : number of function evaluations

    """
    
    q    = 5.                 # Hochschaltbegrenzung       
    rho  = 0.9                # Sicherheitsfaktor           
    s_min = 1.e-12            # absoluter Schwellwert fuer die interne Skalierung
    t0 = t_span[0]            # Startzeitpunkt
    tf = t_span[1]            # Endzeitpunkt
    h_max = (tf - t0)/10.     # Maximale Schrittweite
    h_min = (tf - t0)*1.e-10  # Minimale Schrittweite
    ts = [t0]                 # Liste für die diskrete Zeitpunkte
    ys = [y0]                 # Liste für die diskrete Loesung
    
    t = t0
    y = y0
    h = h_max
    k1 = f(t, y)
    # number of function evaluation
    anzahl_auswertung = 1     
    
    while t < tf:
        t_neu = t + h
        
        # Calculate ki and y according to fomula of Runge Kutta method with adaptive stepsize
        k2 = f(t+h/2, y+h/2*k1)
        k3 = f(t+h/2, y+h/2*k2)
        k4 = f(t+h, y+h*k3)
        y_neu = y +  h/6. *  (k1+2*k2+2*k3+k4)
        k5 = f(t+h, y_neu)

        # every loop has 4 function evaluations
        anzahl_auswertung += 4                 
        
        fehler   = np.linalg.norm(h/6.*(k4 - k5))  # Fehlerschaetzer
        h_opt = h*(rho*tol/fehler)**(1/4)       # Schrittweitenvorschlag
        h_neu     = np.min([h_opt, h_max, q*h])     # Begrenzung der Schrittweite
    
        # Abbruch, falls Schrittweite zu klein
        # necessary to prevent infinite calculation, especially near a point of singularity
        if (h_neu < h_min) :          
            break
        
        # Schritt wird akzeptiert : if error is smaller than tolerance and bigger than minimum acceptable error
        if (fehler <= tol) and (fehler >= s_min) :   
            y  = y_neu
            # FSAL : k1 of y(j+1) is the same with k5 of y(j) 
            k1 = k5.copy()         
            t  = t_neu
            h  = min(h_neu, tf - t)  # damit letzter Zeitschritt richtig
            ys.append(y)             #  an die Liste anhaengen
            ts.append(t)    
        else:                # Schritt wird abgelehnt
            h = h_neu
    
    ys = np.array(ys).transpose()      
    ts = np.array(ts)
    
    # for analysis, number of function evaluaton is returned
    return ODEResult(t=ts, y=ys, auswertung=anzahl_auswertung)  # es sollen dann auch noch 
                                                                # Anzahl Funktionsauswertungen 
                                                                # zurueckgegeben werden


"""    5. test_skalar    """
def test_skalar():
    """
    test rungekutta_43 with scalar example
    """
    # definition of differential equation and excat solution
    def f(t,y):
        return y
    def f_exact(t,t0,y0):
        return np.exp(t+np.log(y0)-t0)
        
    # initial value
    y0 = np.array([1.])
    # interval of t to be dealth
    t_span = [0, 1]
    # values of t to be evaluated
    t_eval = np.linspace(0,1,5)
    
    # solutions by euler forward, runge kutta with adaptive stepsize and exact solution
    sol_euler = euler_forward(f, t_span, y0, t_eval)
    sol_rk = rungekutta_43(f, t_span, y0, tol=1.e-6)
    exact = f_exact(t_eval, t_span[0], y0)
    
    # plotting
    plt.figure(figsize=(8,4))
    plt.rcParams.update({'font.size': 12})
    plt.plot(sol_euler.t, sol_euler.y[0,:], label='euler forward')
    plt.plot(sol_rk.t, sol_rk.y[0,:], 'bo', label='adaptive RK')
    plt.plot(t_eval, exact,  linewidth=3, label='exakt')
    plt.legend()
    # shows the number of evaluation points and function evaluation
    print( len(sol_rk.t) )
    plt.title("Skalar Bsp : Anzahl von t(j):" + str(len(sol_rk.t)) + " and Anzahl der auswertungen :" + str(sol_rk.auswertung))
    plt.show()
    

"""    6. test_system    """
def test_system():
    """
    test rungekutta_43 with vector example
    """
    # definition of differential equation
    def lotka_volterra(t, y, a, b, c, d):
        return np.array( [-a*y[0] + b*y[0]*y[1], c*y[1] - d*y[0]*y[1]] )

    # set of coefficients
    a  = 0.25;
    c  = 1.;
    b  = 0.01;
    d  = 0.05;
    # create the new function with coefficients
    flotka = partial(lotka_volterra, a=a, b=b, c=c, d=d)  

    y0 = [40, 20] # Anfangswert des Systems von DGLen, vektorwertig
    t0 = 0.       # Anfangszeitpunkt
    tf = 60.      # Endzeitpunkt
    t_span = (t0, tf)       # Zeitintervall als Tupel
  
    tk = np.linspace(t0, tf, 1001)  # an diesen Zeitpunkten soll die Loesung berechnet werden 

    # solutions by euler forward, runge kutta with adaptive stepsize and exact solution
    sol_euler = euler_forward(flotka, t_span, y0, t_eval=tk)
    sol_rk = rungekutta_43(flotka, t_span, y0, tol=1.e-6)
    sol_ivp = solve_ivp(flotka, t_span, y0, t_eval=tk, rtol=1.e-9) 
    
    # plotting
    plt.figure(figsize=(16,8))
    plt.rcParams.update({'font.size': 14})
    plt.subplot(1,2,1)
    plt.plot(sol_euler.t, sol_euler.y[0,:], 'r-', label='y1, euler forward')
    plt.plot(sol_euler.t, sol_euler.y[1,:], 'r-', label='y2, euler forward')
    plt.plot(sol_rk.t, sol_rk.y[0,:], 'bo', label='y1, adaptive RK')
    plt.plot(sol_rk.t, sol_rk.y[1,:], 'bo', label='y2, adaptive RK')
    plt.plot(sol_ivp.t, sol_ivp.y[0,:], 'y-', label='y1, ivp solver')
    plt.plot(sol_ivp.t, sol_ivp.y[1,:], 'y-', label='y1, ivp solver')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    

    plt.subplot(1,2,2)
    plt.plot(sol_euler.y[0,:], sol_euler.y[1,:], 'r-', label='euler forward')
    plt.plot(sol_rk.y[0,:], sol_rk.y[1,:], 'bo', label='adaptive RK')
    plt.plot(sol_ivp.y[0,:], sol_ivp.y[1,:], 'y-', label='ivp solver')
    plt.xlabel(r'$y_1$')
    plt.ylabel(r'$y_2$')
    plt.legend()
    # shows the number of evaluation points and function evaluation
    plt.suptitle("Vektorwertig Bsp:  Anzahl von t(j):" + str(len(sol_rk.t)) + " Anzahl der auswertungen :" + str(sol_rk.auswertung))
    plt.show()
    
    
if __name__ == "__main__":
    test_skalar()
    test_system()