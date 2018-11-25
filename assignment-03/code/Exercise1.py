# Exercise 1.1.2

import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fmin
%matplotlib inline

def get_probability_distribution(x, alpha, beta):
    return beta/(np.pi*(beta**2+(x-alpha)**2))

x = np.linspace(-10, 10, num=1000) 
probs = get_probability_distribution(x, 3, 1) 
plt.xlabel(r'$x_k$')
plt.ylabel(r'$p(x_k|\alpha, \beta)$')
plt.plot(x, probs)
plt.savefig('1_1_2.png')
plt.show()

# Exercise 1.1.4

# 10 points in km
data = np.array([3.6, 7.7, -2.6, 4.9, -2.3, 0.2, -7.3, 4.4, 7.3, -5.7])
# distance beta from the shore is known to be 2 km
beta = 2
alpha_list = np.linspace(-10, 10, num=1000)

def get_probability_1_1_4(x, alpha, beta):
    return np.product(beta/(np.pi*beta**2+(x-alpha)**2))


likelihoods = [get_probability_1_1_4(data, alpha, beta) for alpha in alpha_list]
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$p(\alpha|D, \beta = 2)$')
plt.plot(alpha_list, likelihoods)
plt.savefig('1_1_4.png')
plt.show()

print("Mean of the data: {}".format(np.mean(data)))

max_likelihood_index = np.argmax(likelihoods)
max_likelihood_value = np.max(likelihoods)

print("Max likelihood value {} at position {}.".format(max_likelihood_value, max_likelihood_index))
print("Alpha value which maximizes likelihood: {}".format(alpha_list[max_likelihood_index]))



# Exercise 1.2.1

alpha_t = np.random.uniform(0, 10) 
beta_t = np.random.uniform(2, 4)

print(alpha_t)
print(beta_t)



# Exercise 1.2.2
beta_t = 2.436
alpha_t = 5.022

def position(angle, alpha, beta): 
    return beta * np.tan(angle) + alpha

N_points = 500
angles = np.random.uniform(-np.pi/2, np.pi/2, N_points)
positions = [position(angle, alpha_t, beta_t) for angle in angles]

print(positions)



# Exercise 1.2.3

mus = [np.mean(positions[:i + 1]) for i in range(N_points)]
mean = [np.mean(positions)] * (N_points)
X = np.arange(1, N_points + 1)
plt.plot(X, mus, label='Mean over time')
plt.plot(X, mean, label='True mean')
plt.xlabel('data points')
plt.ylabel(r'$\alpha$')
plt.legend()
plt.savefig('1_2_3.png', bbox_inches='tight', dpi=300)
plt.show()

print("Mean of the data points: {}".format(np.mean(positions)))



# Exercise 1.3.2

arrived_datapoints = [1, 2, 3, 20]
alpha_list, beta_list = np.meshgrid(np.linspace(-10, 10, num=500), np.linspace(0, 5, num=250))
plt.style.use('classic')

for k in arrived_datapoints:
    x = positions[:k]
    likelihood = k * np.log(beta_list/np.pi)
    for loc in x:
        likelihood -= np.log(beta_list**2 + (loc - alpha_list)**2)
        
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(alpha_list, beta_list, likelihood, cmap=plt.cm.viridis, vmin=-200, vmax=likelihood.max())
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
    ax.set_zlabel('$\ln p(D | \alpha, \beta)$')
    plt.title('Log likelihood for $k = {}$'.format(k))
    plt.savefig('1_3_2_{}.png'.format(k), bbox_inches='tight', dpi=300)
    plt.show()
    
    
    
    
# Exercise 1.3.3

def log_likelihood(params, positions):
    alpha, beta = params
    likelihood = len(positions) * np.log(beta/np.pi)
    for loc in positions:
        likelihood -= np.log(beta**2 + (loc - alpha)**2)
    return -likelihood
    
def plot_maximize_log_likelihood(data, alpha_t, beta_t):
    alpha_list, beta_list = [], []
    x = np.arange(len(data))
    for k in x:
        [alpha, beta] = fmin(log_likelihood, (0, 1), args=(data[:k],))
        alpha_list.append(alpha)
        beta_list.append(beta)
    
    plt.style.use('ggplot')
    plt.plot(x,alpha_list,label=r'$\alpha$')
    plt.plot(x,beta_list,label=r'$\beta$')
    plt.plot(x,[alpha_t]*len(data),label=r'$\alpha_t$')
    plt.plot(x,[beta_t]*len(data),label=r'$\beta_t$')
    plt.xlabel('$k$')
    plt.ylabel('positions')
    plt.legend()
    plt.savefig('1_3_3.png',bbox_inches='tight',dpi=300)
    plt.show()

    print(alpha_list[-1], beta_list[-1])
    
    
    
    
plot_maximize_log_likelihood(positions, alpha_t, beta_t)    