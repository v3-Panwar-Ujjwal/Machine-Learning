import numpy as np

def find_error(m, c, data_points):
    """
    The functions returns the error of the line from the data points by assuming the line to be of equation
    y = mx+c.
    """
    error = 0.0
    for i in range(len(data_points)):
        x = data_points[i,0]
        y = data_points[i,1]
        error += (y - (m*x +c))**2

    return error/len(data_points)

def gradient_descent_start(initial_m, initial_c, learning_rate, num_iterations, data_points):
    """
    Returns the values of slope and intercept learned.
    Runs gradient descent based on the learning_rate, num_iterations, data and intial slope and intercept value of the line.
    """

    c = initial_c
    m = initial_m

    for i in range((num_iterations)):
        c,m = step_gradient(m, c,np.array(data_points),learning_rate)

    return (c,m)

def step_gradient(m_before, c_before, data_points, learning_rate):
    """
    Calculates the slope and intercept values for the line for each gradient.
    """
    c_gradient = 0
    m_gradient = 0
    n = float(len(data_points))

    for i in range(len(data_points)):
        x = data_points[i,0]
        y = data_points[i,1]

        # Partial derivates to calculate the slope
        c_gradient += (-2/n)*(y - (m_before*x + c_before))
        m_gradient += (-2/n)*x*(y - (m_before*x +c_before))
    
    c = c_before - (c_gradient * learning_rate)
    m = m_before - (m_gradient * learning_rate)
    
    return (c,m)


def start_model():
    # Getting the data
    data_points = np.genfromtxt('data.csv', delimiter=',')
    
    # Initialising the hyperparameter
    learning_rate = 0.00001
    num_iterations = 10000
    initial_m = 0
    initial_c = 0
    
    print("Initial value of slope is: {}, Initial value of c is: {}, The intial error is: {}".format(initial_m, initial_c, find_error(initial_m, initial_c, data_points)))

    print("Running...............")
    c,m = gradient_descent_start(initial_m, initial_c, learning_rate, num_iterations, data_points)

    print("Final value of slope is: {}, Final value of c is: {}, The error is: {}".format(m, c, find_error(m, c, data_points)))

if __name__ == '__main__':
    start_model()