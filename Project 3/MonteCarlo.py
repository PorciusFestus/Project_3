# **************************************************************************************************
# APMA 3100 Project 3       Bruce Bui, Brendan Grimes                                      11/15/21
# **************************************************************************************************

import matplotlib.pyplot as plt
import numpy as np

# *************** Main MonteCarlo Algorithm ****************

# At this point, if you haven't started on this yet, there is still nothing stopping us from using
# either python or java for this, so whichever you prefer. If you use Java, we can just import
# data from a .csv file for analysis, etc. That's what I did for project 2 using R for analysis
# instead of python (python does most everything R can do though, they are comparable for
# analysis purposes)

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


# 1.3 Graph of Three Circles

p = np.array([0.5, 0.7, 0.9])                   # the probabilities that will give the radii
x_p = np.sqrt(-2 * np.log(-1*p + 1) / a ** 2)   # the radii for the three circles

t = [0, 0]
print('The values of x_p are:')
print('for p = ' + str(p[0]) + ', x_p = ' + str(x_p[0]) + ' inches')
print('for p = ' + str(p[1]) + ', x_p = ' + str(x_p[1]) + ' inches')
print('for p = ' + str(p[2]) + ', x_p = ' + str(x_p[2]) + ' inches')

theta = np.linspace(0, 2*np.pi, 100)

c1_x = x_p[0]*np.cos(theta)
c1_y = x_p[0]*np.sin(theta)

c2_x = x_p[1]*np.cos(theta)
c2_y = x_p[1]*np.sin(theta)

c3_x = x_p[2]*np.cos(theta)
c3_y = x_p[2]*np.sin(theta)

fig, ax = plt.subplots(1)

ax.plot(c1_x, c1_y, label='Landing Area within 50% certainty')
ax.plot(c2_x, c2_y, label='Landing Area within 70% certainty')
ax.plot(c3_x, c3_y, label='Landing Area within 90% certainty')
ax.plot(t[0], t[1], 'ro', label='Origin', linewidth=5)
ax.set_aspect(1)

plt.xlabel('East-West Distance')
plt.ylabel('North-South Distance')
plt.title('Landing Areas with Respective Probabilities', fontsize=12)
plt.xlim(-175, 175)
plt.ylim(-250, 150)
plt.grid(linestyle='--')
plt.savefig("three_circles.png", bbox_inches='tight')
plt.legend()
plt.show()



