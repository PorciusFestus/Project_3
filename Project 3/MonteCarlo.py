# **************************************************************************************************
# APMA 3100 Project 3       Bruce Bui, Brendan Grimes                                      11/15/21
# **************************************************************************************************

import matplotlib.pyplot as plt
import numpy as np
from statistics import NormalDist
import pandas as pd

# 1) and 2) are irrelevant to the code, being the assignment and problem statements respectively

# 3) MODEL ANALYSIS
# OBJECTIVE 1: Gain an understanding of the drop error model.

tau = 57  # Standard Deviation of drop point in inches
a = 1 / tau  # Scale Parameter

x = np.linspace(0, 350, 1000)  # Axes Values

# 3.1) Graph of f_X

pdf = a ** 2 * x * np.e ** (-.5 * a ** 2 * x ** 2)
plt.plot(x, pdf, label='f_X (x)')  # Plot some data on the (implicit) axes.
plt.xlabel('Drop Distance x (inches)')
plt.ylabel('Probability Density')
plt.xlim(0, x.max() * 1.1)
plt.ylim(0, pdf.max() * 1.1)
plt.title("PDF of X")
plt.legend()
plt.savefig("pdf_X.png", bbox_inches='tight')
plt.show()

# 3.2) Graph of F_X

cdf = 1 - np.e ** (-.5 * a ** 2 * x ** 2)
plt.plot(x, cdf, label='F_X (x)')  # Plot some data on the (implicit) axes.
plt.xlabel('Drop Distance x (inches)')
plt.ylabel('Cumulative Probability')
plt.xlim(0, x.max() * 1.1)
plt.ylim(0, 1)
plt.title("CDF of X")
plt.legend()
plt.savefig("CDF_X.png", bbox_inches='tight')
plt.show()

# 3.3) Graph of Three Circles

p = np.array([0.5, 0.7, 0.9])  # the probabilities that will give the radii
x_p = np.sqrt(-2 * np.log(-1 * p + 1) / a ** 2)  # the radii for the three circles

t = [0, 0]
print('The values of x_p are:')
print('for p = ' + str(p[0]) + ', x_p = ' + str(x_p[0]) + ' inches')
print('for p = ' + str(p[1]) + ', x_p = ' + str(x_p[1]) + ' inches')
print('for p = ' + str(p[2]) + ', x_p = ' + str(x_p[2]) + ' inches')

theta = np.linspace(0, 2 * np.pi, 100)

c1_x = x_p[0] * np.cos(theta)
c1_y = x_p[0] * np.sin(theta)

c2_x = x_p[1] * np.cos(theta)
c2_y = x_p[1] * np.sin(theta)

c3_x = x_p[2] * np.cos(theta)
c3_y = x_p[2] * np.sin(theta)

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
plt.legend()
plt.savefig("three_circles.png", bbox_inches='tight')
plt.show()

# 4) LAWS OF LARGE NUMBERS

# OBJECTIVE 2: Demonstrate empirically the convergence of the sample mean to the population mean,
# when the sample size increases without bound

# 4.2) Experiment

# 4.2.1) Design a Monte-Carlo simulation algorithm

# The below values are subscripted with 'q' to delineate them from the other previously declared
# variables having the same names.

x_0 = 1000
a_q = 24693
c_q = 3967
K_q = 2 ** 18


# Random Number generator


def next_random():
    global x_0
    x_0 = (a_q * x_0 + c_q) % K_q
    return x_0 / K_q


# inverse cdf of x function


def inv_cdf_x(num):
    return np.sqrt(-2 * np.log(1 - num)) / a


# Monte Carlo Simulation function. Returns a list with n realizations of X.


def monte_carlo(n):
    realizations = []
    for i in range(n):
        realizations.append(inv_cdf_x(next_random()))

    return realizations


# Random Number Generator test

random_nums = []

for i in range(1, 55):
    random_nums.append(round(next_random(), 4))

print("Values of u_1, u_2, and u_3:")
print(random_nums[0])
print(random_nums[1])
print(random_nums[2])
print("Values of u_51, u_52, and u_53:")
print(random_nums[50])
print(random_nums[51])
print(random_nums[52])

# 4.2.2) Simulate the outcomes of many deliveries of newspaper via a drone by generating
# independent realizations of X

# 4.2.3) Calculate 110 independent estimates of the sample mean for the seven sample sizes

# 4.2.4) Make a graph of all the estimates

n_vec1 = [10, 30, 50, 100, 250, 500, 1000]
abscissa1 = []
ordinate1 = []


def sample_mean(count):
    return np.sum(monte_carlo(count)) / count


for n in n_vec1:
    for index in range(110):
        abscissa1.append(n)
        ordinate1.append(sample_mean(n))

mu_x = 1 / a * np.sqrt(np.pi / 2)
mu_x_vec = np.full((770,), mu_x)
var_x = (4 - np.pi) / (2 * a ** 2)

fig, ax0 = plt.subplots(1)

ax0.plot(abscissa1, ordinate1, 'ro', label='M_n')
ax0.plot(abscissa1, mu_x_vec, label='Population Mean')

plt.xlabel('Sample Size (n)')
plt.ylabel('Sample Mean (M_n)')
plt.title('Sample Mean vs n', fontsize=12)
# plt.xlim(-175, 175)
# plt.ylim(-250, 150)
plt.grid(linestyle='--')
plt.legend()
plt.savefig("M_n.png", bbox_inches='tight')
plt.show()

# 4.2.5,6,7 may not require any coding, but if they do, they can go here.

# 4.2.7 TODO: do this thing!

# 5) CENTRAL LIMIT THEOREM

# OBJECTIVE 3: Demonstrate empirically the convergence of Z_n to Z in distribution.

# 5.2) Experiment

# 5.2.1) Prepare samples for the analyses and
# 5.2.2) Perform Analyses

K = 550

# construct the standard normal cdf and vectors for graphing it.
normal = NormalDist()
phi_abscissa = np.linspace(-2.5, 2.5, K)
phi = []
for phi_i in phi_abscissa:
    phi.append(normal.cdf(phi_i))

# Variables to store quantities for 5.3
mean_var_n = []
AD_n = []
n_vec2 = np.array([5, 10, 15, 30])


n_index = 1

for n in n_vec2:
    abscissa2 = []
    ordinate2 = []
    m_n = []
    z_n = []
    F_n = []
    MAD_x = 0
    MAD_y = 0
    MAD_index = 0

    # Prepare samples (5.2.1)
    for k in range(K):
        m_n.append(sample_mean(n))

    # 5.2.2.1) calculate the estimates of the mean and variance of M_n
    mean = np.sum(m_n) / K
    variance = np.sum(np.subtract(np.power(m_n, 2), mean ** 2)) / K

    mean_var_n.append(np.round([n, mean, np.sqrt(variance), mu_x, np.sqrt(var_x) / np.sqrt(n)], 4))

    # 5.2.2.2) transform the sample of M_n into a sample of the standardized random variable Z_n
    z_n = np.divide(np.subtract(m_n, mean), np.sqrt(variance))

    # 5.2.2.3) estimate from the sample of Z_n the probabilities of seven events
    z_j = np.array([-1.4, -1.0, -0.5, 0, 0.5, 1.0, 1.4])

    AD_j = [n]
    for j in range(z_j.size):
        # find proportion of z_n values less than or equal to z_j
        counter = 0
        for z in z_n:
            if z <= z_j[j]:
                counter += 1

        F_n.append(counter / K)

        # 5.2.2.4) Evaluate the goodness-of-fit using the MAD
        MAD_j = abs(F_n[j] - normal.cdf(z_j[j]))
        AD_j.append(MAD_j)

        if MAD_j > MAD_y:
            MAD_y = MAD_j
            MAD_x = z_j[j]
            MAD_index = j
    AD_j.append(np.round(MAD_y, 4))
    AD_n.append(np.round(AD_j, 4))

    # 5.2.2.5) Draw a figure showing:

    plt.subplot(np.ceil(np.sqrt(n_vec2.size)), np.ceil(np.sqrt(n_vec2.size)), n_index)

    # i) the seven points {(z_j, F_n(z_j)) : j = 1, ... ,7}
    plt.scatter(z_j, F_n, label='F_n(z_j)', color='cyan', linewidth=0.25 )

    # ii) The standard normal cdf phi
    plt.plot(phi_abscissa, phi, label='Phi(z)')

    # iii) The MAD as a highlighted interval of probability (ordinate at point z_j at which it
    # occurs (abscissa)
    plt.vlines(x=MAD_x, ymin=min(normal.cdf(MAD_x), F_n[MAD_index]),
               ymax=max(normal.cdf(MAD_x), F_n[MAD_index]), colors='red', label='MAD_n')

    # And make it look nice :)
    plt.xlabel('Z')
    plt.ylabel('Cumulative Probability')
    nameString = 'Graph_of_Fn-n' + str(n) + '.png'
    plt.title(nameString, fontsize=12)
    # plt.xlim(-175, 175)
    # plt.ylim(-250, 150)
    plt.grid(linestyle='--')
    plt.legend()
    plt.savefig(nameString, bbox_inches='tight')

    n_index += 1

plt.show()

# plt.savefig('5.3_Graph_Panel.png', bbox_inches='tight')

# 5.3) Summarize Results
# NOTE: LaTeX will read from a .csv file, so we probably want the code to output tabular results
# as .csv files which we can then read directly instead of typing out the tables by hand,
# which can be somewhat tedious, especially for large tables.

# The data were collected in previous areas of the code, so now we need only export it to a .csv

# i) is the already saved graphs

# ii) a table comparing the estimates (mu_n, sigma_n) with the population values
# (mu_x, sigma_x / sqrt(n) ) for every n.

mean_var_n.insert(0, ['n', 'Sample Mean', 'Sample Standard Deviation', 'Population Mean',
                      'Population Standard Deviation'])
mean_var_df = pd.DataFrame(mean_var_n)
# mean_var_df.drop(index=mean_var_df.index[0], axis=0, inplace=True)
mean_var_df.to_csv('mean_var.csv', index=False)

# iii) a table reporting the absolute difference for every j and n, and the MAD for every n
AD_n.insert(0, np.array(['n', 'AD j = 1', 'AD j = 2',
                         'AD j = 3', 'AD j = 4',
                         'AD j = 5', 'AD j = 6',
                         'AD j = 7', 'MAD_n']))
AD_df = pd.DataFrame(AD_n)
# AD_df.drop(index=AD_df.index[0], axis=0, inplace=True)
AD_df.to_csv('AD_n.csv', index=False)

# 5.4) and 5.5) should be strictly report.
