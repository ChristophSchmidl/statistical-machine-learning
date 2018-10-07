'''
*** Authors ***

Name: Christoph Schmidl
Studentnumber: s4226887
Studentemail: c.schmidl@student.ru.nl

Name: Mark Beijer
Studentnumber: s4354834
Studentemail: mbeijer@science.ru.nl

'''

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
# Return evenly spaced numbers over a specified interval.

# Exercise 1.1

def f(x):
	return 1 + np.sin(6*(x-2))

def noisy_function(func, func_argument):
	noise = np.random.normal(0, 0.3)
	return noise + func(func_argument)

def generate_data(amount_of_datapoints):
    return [noisy_function(f, x) for x in np.linspace(0, 1, amount_of_datapoints)]

def plot_data(dataset):
    # plot the dataset
    plt.scatter(np.linspace(0, 1, 10), dataset, label='noisy observations')
    # plot the actual function
    X = np.linspace(0, 1, 100) # the higher the num value, the smoother the function plot gets
    y = [f(x) for x in X]
    plt.plot(X, y, color='green', label='True function')
    # plt.xlim(xmin=0)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend(loc='upper right')
    # fancy caption. Not needed if latex is doing the job later on
    #txt="I need the caption to be present a little below X-axis"
    #plt.figtext(0.5, -0.05, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig('exercise_1_1.png')
    plt.show()

# Generating data set D_10 of 10 noisy observations
training_set = generate_data(10)

# Generating data set T of 100 noisy observations
test_set = generate_data(100)

# Plotting the function and observations in D_10
plot_data(training_set)



# Exercise 1.2

def SSE(observations, targets):
    """ Calculate the sum-of-squares error. """
    return 0.5 * np.sum((observations - targets)**2)


def pol_cur_fit(data, polynomial_order):
    """ Return weights for an optimal polynomial curve fit. """
    
    polynomial_order = polynomial_order + 1
    
    observations = data[0, :] # Get me the first row, D_N
    targets = data[1, :] # Get me the second row, M
    
    # observation matrix
    A = np.zeros((polynomial_order, polynomial_order)) # Create matrix
    for i in range(polynomial_order):
        for j in range(polynomial_order):
            A[i, j] = np.sum(observations ** (i+j))
    
    # target vector        
    B = np.zeros(polynomial_order)
    for i in range(polynomial_order):
        B[i] = np.sum(targets * observations**i)
    
    # numpy.linalg.solve(a, b)
    # Solve a linear matrix equation, or system of linear scalar equations.
    # Computes the “exact” solution, x, of the well-determined, i.e., full rank, linear matrix equation ax = b.
    
    # Here's where the magic happens. Solve the linear system.
    weights = np.linalg.solve(A, B)
    return weights



# Exercise 1.3

def evaluate_polynomial(point, weights):
    """ Evaluate a polynomial. """
    return np.polyval(list(reversed(weights)), point)

def RMSE(observations, targets):
    """ Calculate the root-mean-squared error. """
    error = SSE(observations, targets)
    return np.sqrt(2 * error / len(observations))


def evaluate_and_plot_curve_fitting(training_set, test_set, min_polynomial_order = 0, max_polynomial_order = 10):
    """Evaluate the RMSE based on different polynomial orders"""
    
    errors_train = []
    errors_test = []
    
    X = np.linspace(0, 1, 100)
    y = [f(x) for x in X]
    
    for m in range(min_polynomial_order, max_polynomial_order):
        w = pol_cur_fit(training_data, m)
        fitted_curve = evaluate_polynomial(X, w)
        
        print("Evaluated polynomial: {}".format(fitted_curve))

        rmse_train = RMSE(evaluate_polynomial(training_set[0, :], w), training_set[1, :])
        rmse_test = RMSE(evaluate_polynomial(test_set[0, :], w), test_set[1, :])
        errors_train.append(rmse_train)
        errors_test.append(rmse_test)
        
        plt.figure()
        plt.plot(X, fitted_curve, 'b', label='Fitted Curve')
        plt.plot(X, y, 'r', label='True function')
        plt.scatter(training_set[0, :], training_set[1, :], c='g', label='Noisy observations')
        plt.title('M=%d: train RMSE=%.2f, test RMSE=%.2f' % (m, rmse_train, rmse_test))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.savefig('curvefit_m%d_n_%d.png' % (m, training_set.shape[1]))
        plt.show()
        
    plt.figure()    
    x_range = np.arange(min_polynomial_order, max_polynomial_order)
        
    plt.plot(x_range, errors_test, 'r', label='Test')
    plt.plot(x_range, errors_train, 'b', label='Training')
    #plt.ylim([0, 0.4])
    plt.legend()
    plt.xlabel("Polynomial order")
    plt.ylabel("RMSE")
    plt.savefig("rmse_polynomial_order.png")
    plt.show()
    
    print(errors_train)
    print(errors_test)
  
        
# Add the linear spacing of y values to training and test set
full_training_set = np.vstack((np.linspace(0, 1, 10), training_set))
full_test_set = np.vstack((np.linspace(0, 1, 100), test_set))        
        
evaluate_and_plot_curve_fitting(full_training_set, full_test_set)    




# Generating data set D_40 of 40 noisy observations
training_set_40 = generate_data(40)
full_training_set_40 = np.vstack((np.linspace(0, 1, 40), training_set_40))

evaluate_and_plot_curve_fitting(full_training_set_40, full_test_set)






def pol_cur_fit_with_regularization(data, polynomial_order, regularizer = 0.00):
    """ Return weights for an optimal polynomial curve fit. """
    
    observations = data[0, :] # Get me the first row, D_N
    targets = data[1, :] # Get me the second row, M
    
    # observation matrix
    A = np.zeros((polynomial_order, polynomial_order)) # Create matrix
    for i in range(polynomial_order):
        for j in range(polynomial_order):
            A[i, j] = np.sum(observations ** (i+j))
    
    # Regularization
    # Multiply the diagonal matrix of order m with regularization term and add it to A
    A = A + ( regularizer * np.identity(polynomial_order) )
    
    # target vector        
    B = np.zeros(polynomial_order)
    for i in range(polynomial_order):
        B[i] = np.sum(targets * observations**i)
    
    # numpy.linalg.solve(a, b)
    # Solve a linear matrix equation, or system of linear scalar equations.
    # Computes the “exact” solution, x, of the well-determined, i.e., full rank, linear matrix equation ax = b.
    
    # Here's where the magic happens. Solve the linear system.
    weights = np.linalg.solve(A, B)
    return weights


def evaluate_and_plot_curve_fitting_with_reg(training_set, 
                                             test_set,
                                             reg = 0.1):
    """Evaluate the RMSE based on different polynomial orders"""
    
    errors_train = []
    errors_test = []
    
    regularizer_range = np.arange(-40, -20)
    exp_regularizer_range = np.exp(regularizer_range) # perform e^x because Bishop uses ln lambda
    
    for regularizer_value in exp_regularizer_range:
        w = pol_cur_fit_with_regularization(training_set, 9, regularizer_value) # fix polynomial order to 9
        rmse_train = RMSE(evaluate_polynomial(training_set[0, :], w), training_data[1, :])
        rmse_test = RMSE(evaluate_polynomial(test_set[0, :], w), test_set[1, :])
        errors_train.append(rmse_train)
        errors_test.append(rmse_test)
        
    plt.figure()    
        
    plt.plot(regularizer_range, errors_test, 'r', label='Test')
    plt.plot(regularizer_range, errors_train, 'b', label='Training')
    #plt.ylim([0, 0.4])
    plt.legend()
    plt.xlabel(r'$\ln {\lambda}$')
    plt.ylabel("RMSE")
    plt.savefig("rmse_polynomial_order_reg.png")
    plt.show()
        
        
weights = w = pol_cur_fit(training_data, 9)
weights_regularized = pol_cur_fit_with_regularization(training_data, 9, 0.1)

print(weights)
print(weights_regularized)
        
evaluate_and_plot_curve_fitting_with_reg(full_training_set, full_test_set)

