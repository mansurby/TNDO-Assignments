# TNDO Home Work
"""
Numerical Methods Comparison
Problem: dy/dx = x + y^2, y(0) = 1
Find: y(0.2) using h = 0.1
Methods: Euler, RK2, RK4, RK45
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def f(x, y):
    """Differential equation: dy/dx = x + y^2"""
    return x + y**2


def euler(x0, y0, h, x_end):
    """Euler's method"""
    x = x0
    y = y0
    xs, ys = [x0], [y0]

    while x < x_end:
        y = y + h * f(x, y)
        x = x + h
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


def rk2(x0, y0, h, x_end):
    """2nd order Runge-Kutta (midpoint method)"""
    x = x0
    y = y0
    xs, ys = [x0], [y0]

    while x < x_end:
        k1 = f(x, y)
        k2 = f(x + h/2, y + h*k1/2)
        y = y + h * k2
        x = x + h
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


def rk4(x0, y0, h, x_end):
    """4th order Runge-Kutta method"""
    x = x0
    y = y0
    xs, ys = [x0], [y0]

    while x < x_end:
        k1 = f(x, y)
        k2 = f(x + h/2, y + h*k1/2)
        k3 = f(x + h/2, y + h*k2/2)
        k4 = f(x + h, y + h*k3)
        y = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        x = x + h
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


def rk45(x0, y0, h, x_end, tol=1e-6):
    """Adaptive Runge-Kutta 4/5 method"""
    x = x0
    y = y0
    xs, ys = [x0], [y0]

    while x < x_end:
        if x + h > x_end:
            h = x_end - x

        # Calculate k values
        k1 = h * f(x, y)
        k2 = h * f(x + h/5, y + k1/5)
        k3 = h * f(x + 3*h/10, y + 3*k1/40 + 9*k2/40)
        k4 = h * f(x + 3*h/5, y + 3*k1/10 - 9*k2/10 + 6*k3/5)
        k5 = h * f(x + h, y - 11*k1/54 + 5*k2/2 - 70*k3/27 + 35*k4/27)
        k6 = h * f(x + 7*h/8, y + 1631*k1/55296 + 175*k2/512 +
                   575*k3/13824 + 44275*k4/110592 + 253*k5/4096)

        # 4th and 5th order estimates
        y4 = y + 37*k1/378 + 250*k3/621 + 125*k4/594 + 512*k6/1771
        y5 = y + 2825*k1/27648 + 18575*k3/48384 + 13525*k4/55296 + 277*k5/14336 + k6/4

        # Error estimate
        error = abs(y5 - y4)

        if error <= tol or h < 1e-10:
            x += h
            y = y5
            xs.append(x)
            ys.append(y)

        # Adjust step size
        if error > 0:
            h = h * min(2.0, max(0.5, 0.9 * (tol / error)**0.2))
        else:
            h = h * 2.0

    return np.array(xs), np.array(ys)


def reference_solution(x0, y0, x_end):
    """High-precision reference using very small RK4 steps"""
    h = 1e-6
    x, y = x0, y0

    while x < x_end:
        if x + h > x_end:
            h = x_end - x

        k1 = f(x, y)
        k2 = f(x + h/2, y + h*k1/2)
        k3 = f(x + h/2, y + h*k2/2)
        k4 = f(x + h, y + h*k3)
        y += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h

    return y


def main():
    # Problem parameters
    x0, y0, h, x_target = 0, 1, 0.1, 0.2

    print("Numerical Methods Comparison")
    print("=" * 40)
    print(f"Problem: dy/dx = x + y², y(0) = 1")
    print(f"Target: y({x_target}) with step size h = {h}")
    print("=" * 40)

    # Define methods
    methods = {
        "Euler": euler,
        "RK2": rk2,
        "RK4": rk4,
        "RK45": rk45
    }

    # Run all methods and collect results
    results = {}
    times = {}

    for name, method in methods.items():
        print(f"Running {name}...")
        start_time = time.time()

        try:
            xs, ys = method(x0, y0, h, x_target)
            end_time = time.time()

            results[name] = (xs, ys)
            times[name] = end_time - start_time
            print(f"  ✓ {name}: y({x_target}) = {ys[-1]:.8f}")

        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            results[name] = (np.array([x0]), np.array([y0]))
            times[name] = 0

    # Calculate reference solution
    print("\nCalculating reference solution...")
    y_ref = reference_solution(x0, y0, x_target)
    print(f"Reference: y({x_target}) = {y_ref:.10f}")

    # Print results table
    print(f"\nResults Summary")
    print("-" * 55)
    print(f"{'Method':<8} {'y(0.2)':<12} {'Error':<12} {'Time(s)':<8}")
    print("-" * 55)

    for name in methods.keys():
        if name in results and len(results[name][1]) > 1:
            y_final = results[name][1][-1]
            error = abs(y_final - y_ref)
            time_taken = times[name]
            print(f"{name:<8} {y_final:<12.6f} {error:<12.2e} {time_taken:<8.4f}")

    # Create visualization
    plt.figure(figsize=(12, 5))

    # Plot 1: Solution curves
    plt.subplot(1, 2, 1)
    colors = ['red', 'green', 'blue', 'orange']

    for i, (name, (xs, ys)) in enumerate(results.items()):
        if len(xs) > 1:
            plt.plot(xs, ys, 'o-', color=colors[i],
                     label=name, linewidth=2, markersize=4)

    plt.title("Solution Comparison")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Error comparison
    plt.subplot(1, 2, 2)
    method_names = []
    errors = []

    for name, (xs, ys) in results.items():
        if len(ys) > 1:
            error = abs(ys[-1] - y_ref)
            method_names.append(name)
            errors.append(error)

    bars = plt.bar(method_names, errors,
                   color=colors[:len(method_names)], alpha=0.7)
    plt.title("Final Errors at x = 0.2")
    plt.ylabel("Absolute Error")
    plt.yscale('log')
    plt.grid(True, alpha=0.3, axis='y')

    # Add error values on bars
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 f'{error:.1e}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    # Summary
    print(f"\nSummary:")
    if method_names and errors:
        best_method = method_names[errors.index(min(errors))]
        fastest_method = min(
            times.keys(), key=lambda k: times[k] if times[k] > 0 else float('inf'))
        print(f"Most accurate: {best_method}")
        print(f"Fastest: {fastest_method}")


if __name__ == "__main__":
    main()
