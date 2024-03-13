# Parabolic-Partial-Differential-Equation
The goal of this project is to find the numerical solution of a parabolic partial differential equation using the method of lines. This parabolic partial differential equation is a stiff problem. I and Suh worked on this project together.

# Project Description
The project is organized in two parts:

**1. Implementation of the Forward and Backward Euler Methods:**

The forward and backward Euler methods are implemented in _odesolver.py_. These methods are used to solve a parabolic partial differential equation.

**2. Report of the Project:**

The report of this poject is written in _Report-Parabolic-Partial-Diffential-Equation.ipynb_. The relation between stiffness and eigenvalues is explained, the result of the experiment with forward(explicit) and backward(implicit) Euler Methods are reported, and in addition the convection-diffusion equation is solved.

# Contents
The repository contains the following:

- _odesolver.py_: Python script containing the implementation of numerical methods to solve differential equations. Not only Euler methods but also Runge-Kutta methods are implemented.
- _Report-Parabolic-Partial-Diffential-Equation.ipynb_: Jupyter Notebook that uses the implemented methods in _odesolver.py_, explains the theory and reports the result.
- _README.md_: The file you are reading now.
