# Numerical Methods for Solving ODEs in Orbital Dynamics

## Project Description
This project applies numerical methods to solve the ordinary differential equation:
$$
\frac{dy}{dx} = x + y^2
$$
The goal is to approximate the value of \(y\) at \(x = 0.2\) given the initial condition \(y(0) = 1\), using a step size of \(h = 0.1\).
The numerical methods implemented include:
- Euler’s Method
- Heun’s Method (RK2)
- Runge-Kutta-Potts-Oda Method (RK4 variant)
- Runge-Kutta 4-5 Method (RK45)
This allows comparison of the accuracy and efficiency of each method in solving nonlinear ODEs in orbital dynamics.
## Usage
1. Clone the repository:
https://github.com/mansurby/TNDO-Assignments.git
2. Open the scripts in Python and run the method of your choice.
3. Check the results in the output files or console.

## Results
The code approximates \(y\) at \(x = 0.2\) for each method. You can compare the results to see how step size and method affect accuracy.

pip install -r requirements.txt

python RK_45.py

