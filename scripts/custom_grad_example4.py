#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/

# Custom gradient example with vector (signal) and vector (of parameter) arguments
# f(x, y) = x^2 * prod(y), where x and y are a vector
import numpy as np
import tensorflow as tf

# Python function with no custom gradient
def foo_no_grad(x, y):
    y = np.square(x)*np.prod(y)
    return tf.constant(y)


# Python function with custom analytic gradient
@tf.custom_gradient
def foo_custom_grad_analytic(x, y):
    
    z = np.square(x)*np.prod(y)
 
    def grad_fn(dy):
        
        # Gradient w.r.t x
        gradx = 2 * np.array(x) * np.prod(y)
        
        # Gradient w.r.t y
        vecJ = np.zeros_like(y)
        for i in range(y.shape[0]):
            vecJ[i] = np.prod(np.delete(y, [i]))*np.dot(np.transpose(dy), np.square(x))
        
        # Returns vector-Jacobian
        return gradx * dy, vecJ

    
    return z, grad_fn


# Python function with custom numerical gradient
@tf.custom_gradient
def foo_custom_grad_numeric(x, y):
    
    epsilon = 0.01
    
    def func(x, y):
        z = np.square(x)*np.prod(y)
        return z
    
    def grad_fn(dy):
        
        # Grad w.r.t x
        J_plus = func(x + epsilon, y)
        J_minus = func(x - epsilon, y)
        gradx = (J_plus -  J_minus)/(2.0*epsilon) 
       
        # Grad w.r.t y
        yc = y.numpy()
        
        # pre-allocate vector * Jaccobian output
        vecJ = np.zeros_like(y)
        
        # Iterate over each parameter and compute the output
        for i in range(y.shape[0]):
            
            yc[i] = yc[i] + epsilon
            J_plus = func(x, yc)
            yc[i] = yc[i] - 2*epsilon
            J_minus = func(x, yc)
            grady = (J_plus -  J_minus)/(2.0*epsilon) 
            yc[i] = yc[i] + 1*epsilon
            vecJ[i] = np.dot(np.transpose(dy), grady)
        
        return gradx * dy, vecJ
    
    return func(x, y), grad_fn

# Python function only using tf ops with gradient automatically defined via auto-diff 
def foo_autodiff(x, y):
    y = tf.math.reduce_prod(y)*tf.square(x)
    return y

with tf.GradientTape(persistent=True) as tape:
    
    a = 2*np.ones((5,1)) # signal
    b = 3*np.ones((3,1))# b[1:2] = 1 # params
    x = tf.constant(a, dtype=tf.float64)
    y = tf.constant(b, dtype=tf.float64)
    tape.watch(x)
    tape.watch(y)
    
    
    z1 = foo_autodiff(x, y)**2
    z2 = foo_no_grad(x,y)**2
    z3 = foo_custom_grad_analytic(x, y)**2
    z4 = foo_custom_grad_numeric(x, y)**2


print('\nGrad w.r.t. x')
print("foo_autodiff", tape.gradient(z3, x))   
print("foo_no_grad", tape.gradient(z1, x)) 
print("foo_custom_grad_analytic", tape.gradient(z2, x))   
print("foo_custom_grad_numeric", tape.gradient(z4, x))  

print('\nGrad w.r.t. y')
print("foo_autodiff", tape.gradient(z1, y))   
print("foo_no_grad", tape.gradient(z2, y)) 
print("foo_custom_grad_analytic", tape.gradient(z3, y))   
print("foo_custom_grad_numeric", tape.gradient(z4, y))   