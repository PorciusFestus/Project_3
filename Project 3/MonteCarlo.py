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
x_p_sq = -2 * np.log(-1*p + 1) / a ** 2         # the SQUARES of the radii for the three circles

t = [0, 0]
print('The values of x_p are:')
print('for p = ' + str(p[0]) + ', x_p = ' + str(np.sqrt(x_p_sq[0])) + ' inches')
print('for p = ' + str(p[1]) + ', x_p = ' + str(np.sqrt(x_p_sq[1])) + ' inches')
print('for p = ' + str(p[2]) + ', x_p = ' + str(np.sqrt(x_p_sq[2])) + ' inches')

x_coord = np.linspace(-1*np.floor(np.sqrt(x_p_sq[2])), np.ceil(np.sqrt(x_p_sq[2])), 1000)
y_coord = np.linspace(-1*np.floor(np.sqrt(x_p_sq[2])), np.ceil(np.sqrt(x_p_sq[2])), 1000)

X_coord, Y_coord = np.meshgrid(x_coord,y_coord)

Circle_1 = X_coord**2 + Y_coord**2 - x_p_sq[0]
Circle_2 = X_coord**2 + Y_coord**2 - x_p_sq[1]
Circle_3 = X_coord**2 + Y_coord**2 - x_p_sq[2]

fig, ax = plt.subplots()

ax.contour(X_coord, Y_coord, Circle_1, [0], label='Landing Area within 50% certainty')
ax.contour(X_coord, Y_coord, Circle_2, [0], label='Landing Area within 70% certainty')
ax.contour(X_coord, Y_coord, Circle_3, [0], label='Landing Area within 90% certainty')
# plt.plot(t[0], t[1], label='Origin', linewidth=5)                # The origin (point T)
ax.set_aspect(1)

plt.xlabel('East-West Distance')
plt.ylabel('North-South Distance')
plt.title('Landing Areas with Respective Probabilities', fontsize=8)
plt.xlim(-1.1*np.floor(np.sqrt(x_p_sq[2])), 1.1*np.ceil(np.sqrt(x_p_sq[2])))
plt.ylim(-1.1*np.floor(np.sqrt(x_p_sq[2])), 1.1*np.ceil(np.sqrt(x_p_sq[2])))
# plt.legend()   <--- Does not seem to work with this plotting method
plt.grid(linestyle='--')

plt.savefig("plot_circle_matplotlib_03.png", bbox_inches='tight')

plt.show()



