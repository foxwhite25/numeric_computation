import numpy as np
import scipy.integrate as spi


def f(x):
    return np.log(x)


exact_integral = 2 * np.log(2) - 1

# 1. The composite Trapezoid Rule with m = 4 panels
x = np.linspace(1, 2, 50)
y = f(x)
trapz_integral = spi.trapz(y, x)
trapz_error = np.abs(trapz_integral - exact_integral)

# 2. The composite Simpson’s Rule with m = 4 panels
simp_integral = spi.simpson(y, x)
simp_error = np.abs(simp_integral - exact_integral)

midpoints = (x[:-1] + x[1:]) / 2
mid_integral = np.sum(f(midpoints) * (x[1] - x[0]))
mid_error = np.abs(mid_integral - exact_integral)

# 4. The Romberg Integration Rule with j = 3
romb_integral = spi.romberg(f, 1, 2, divmax=3)
romb_error = np.abs(romb_integral - exact_integral)

# 5. The Gaussian Quadrature Method with n = 4
gauss_integral, _ = spi.fixed_quad(f, 1, 2, n=4)
gauss_error = np.abs(gauss_integral - exact_integral)

if __name__ == '__main__':
    print(f"Trapezoid Rule: Integral = {trapz_integral}, Error = {trapz_error}")
    print(f"Simpson’s Rule: Integral = {simp_integral}, Error = {simp_error}")
    print(f"Midpoint Rule: Integral = {mid_integral}, Error = {mid_error}")
    print(f"Romberg Integration: Integral = {romb_integral}, Error = {romb_error}")
    print(f"Gaussian Quadrature: Integral = {gauss_integral}, Error = {gauss_error}")
