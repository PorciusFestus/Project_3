# **************************************************************************************************
# APMA 3100 Project 3       Bruce Bui, Brendan Grimes                                      11/15/21
# **************************************************************************************************

import matplotlib.pyplot as plt
import numpy as np

# *************** Main MonteCarlo Algorithm ****************

# 3) Model Analysis
# OBJECTIVE 1: Gain an understanding of th drop error model.

tau = 57                            # Standard Deviation of drop point in inches
a = 1/tau                           # Scale Parameter

x = np.linspace(0, 350, 1000)       # Axes Values

# 1.1: Graph of f_X

pdf = a**2 * x * np.e**(-.5*a**2 * x**2)
plt.plot(x, pdf, label='f_X (x)')  # Plot some data on the (implicit) axes.
plt.xlabel('Drop Distance x')
plt.ylabel('Probability Density')
plt.xlim(0, x.max()*1.1)
plt.ylim(0, pdf.max()*1.1)
plt.title("PDF of X")
plt.legend()
plt.show()

# 1.2 Graph of F_X

cdf = 1 - np.e**(-.5*a**2 * x**2)
plt.plot(x, cdf, label='F_X (x)')  # Plot some data on the (implicit) axes.
plt.xlabel('Drop Distance x')
plt.ylabel('Cumulative Probability')
plt.xlim(0, x.max()*1.1)
plt.ylim(0, 1)
plt.title("CDF of X")
plt.legend()
plt.show()


# 1.3 Graph of Three Circles TODO: this still needs a good bit of work, although the x_p values work

p = np.array([0.5, 0.7, 0.9])                   # the probabilities that will give the radii
x_vals = np.linspace(-130, 130, 1000)           # horizontal axis coordinates
x2_vals = np.array([x_vals, x_vals])            # need to do each x val twice since graphing circles
x_p_sq = -2 * np.log(-1*p + 1) / a ** 2         # the SQUARES of the radii for the three circles

t = [0, 0]
print('The values of x_p are:')
print('for p = ' + str(p[0]) + ', x_p = ' + str(np.sqrt(x_p_sq[0])) + ' inches')
print('for p = ' + str(p[1]) + ', x_p = ' + str(np.sqrt(x_p_sq[1])) + ' inches')
print('for p = ' + str(p[2]) + ', x_p = ' + str(np.sqrt(x_p_sq[2])) + ' inches')

# first circle:
circle_1_neg = -1 * np.sqrt(x_p_sq[0] - x_vals ** 2)
circle_1_pos = np.sqrt(x_p_sq[0] - x_vals ** 2)
circle_1 = [circle_1_neg, circle_1_pos]

# second circle:
circle_2_neg = -1 * np.sqrt(x_p_sq[1] - x_vals ** 2)
circle_2_pos = np.sqrt(x_p_sq[1] - x_vals ** 2)
circle_2 = [circle_1_neg, circle_1_pos]

# third circle:
circle_3_neg = -1 * np.sqrt(x_p_sq[2] - x_vals ** 2)
circle_3_pos = np.sqrt(x_p_sq[2] - x_vals ** 2)
circle_3 = [circle_1_neg, circle_1_pos]

# plot the circle and the point

plt.plot(x_vals, circle_1_pos, label='Landing Area within 50% certainty')  # The first circle
plt.plot(x_vals, circle_2_pos, label='Landing Area within 70% certainty')  # The second circle
plt.plot(x_vals, circle_3_pos, label='Landing Area within 90% certainty')  # The third circle
plt.plot(t[0], t[1], label='Origin', linewidth=5)                      # The origin (point T)
plt.xlabel('East-West Distance')
plt.ylabel('North-South Distance')
plt.xlim(x2_vals.min()*1.1, x2_vals.max()*1.1)
plt.ylim(x2_vals.min()*1.1, x2_vals.max()*1.1)
plt.title("Landing Areas with Respective Probabilities")
# plt.legend()
plt.show()

