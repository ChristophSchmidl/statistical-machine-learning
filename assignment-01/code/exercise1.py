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



def f(x):
	return np.sin(6*(x-2))

def noisy_function(func, func_argument):
	noise = np.random.normal(0, 0.3)
	return noise + func(func_argument)

def generate_data(amount_of_datapoints):
    return [noisy_function(f, x) for x in np.linspace(0, 1, amount_of_datapoints)]
    
training_set = generate_data(10)
test_set = generate_data(100)
plt.scatter(np.linspace(0, 1, 10), training_set)
X = np.linspace(0, 1, 100)
y = [f(x) for x in X]
plt.plot(X, y)
plt.show()