import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D


# Exercise 3.1.1

# numpy.linalg.inv(a)
# Compute the (multiplicative) inverse of a matrix.
# Given a square matrix a, return the matrix ainv 
# satisfying dot(a, ainv) = dot(ainv, a) = eye(a.shape[0]).

# See: https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.linalg.inv.html

covariance_matrix = np.array([
        [0.14, -0.3, 0.0, 0.2], 
        [-0.3, 1.16, 0.2, -0.8], 
        [0.0, 0.2, 1.0, 1.0], 
        [0.2, -0.8, 1.0, 2.0]])

precision_matrix = np.linalg.inv(covariance_matrix)

print(precision_matrix)


precision_matrix_aa = np.array([
        [60, 50], 
        [50, 50]])

inv_presision_matrix_aa = np.linalg.inv(precision_matrix_aa)

print(inv_presision_matrix_aa)


# Exercise 3.1.2

# numpy.random.multivariate_normal(mean, cov[, size, check_valid, tol])

# Draw random samples from a multivariate normal distribution.

# The multivariate normal, multinormal or Gaussian distribution is a 
# generalization of the one-dimensional normal distribution to higher
# dimensions. Such a distribution is specified by its mean and covariance
# matrix. These parameters are analogous to the mean (average or “center”)
# and variance (standard deviation, or “width,” squared) of the 
# one-dimensional normal distribution.

def generate_random_number_pair():
    return np.random.multivariate_normal(
    [0.8, 0.8],
    [[0.1, -0.1], [-0.1, 0.12]])

random_pair = generate_random_number_pair()
print(random_pair)



# Exercise 3.1.3

x, y = np.mgrid[-0.25:2.25:.01, -1:2:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y
mu_p = [0.8, 0.8]
cov_p = [[0.1, -0.1], [-0.1, 0.12]]
z = multivariate_normal(mu_p, cov_p).pdf(pos)

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, z)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
ax.set_zlabel('$p(x_1, x_2 | x_3=0, x_4=0)$')
plt.savefig('3_1_3.png', bbox_inches='tight', dpi=300)
plt.show()



# Exercise 3.2.1

number_of_datapoints = 1000

covariance_3_2 = np.array([[2.0, 0.8],[0.8, 4.0]])
print(covariance_3_2)

data = np.random.multivariate_normal(
    random_pair,
    covariance_3_2,
    number_of_datapoints)

np.savetxt('ex3_data.txt', data)



# Exercise 3.2.2

mle_mean = np.mean(data, axis=0) # to take the mean of each col
print(mean)
normalized_data = data - mle_mean
mle_covariance = np.dot(normalized_data.T, normalized_data) / number_of_datapoints
mle_covariance_unbiased =  np.dot(normalized_data.T, normalized_data) / (number_of_datapoints - 1)
print(mle_covariance)
print(mle_covariance_unbiased)


# Exercise 3.3.1

mu = 0
for i in range(1, np.size(data, 0)+1):
    mu = mu + 1.0 / i * (data[i-1] - mu)
    print(mu)