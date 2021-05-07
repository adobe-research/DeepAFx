#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/

# Custom gradient example with two scalar arguments of the function:
# f(x, y) = x^2 * y

import numpy as np
import tensorflow as tf

# Python function with no custom gradient
def foo_no_grad(x, y):
    y = np.square(x)*y
    return tf.constant(y)


# Python function with custom analytic gradient
@tf.custom_gradient
def foo_custom_grad_analytic(x, y):
    
    z = np.square(x)*y
 
    def grad_fn(dy):
        grad = 2 * np.array(x) * y
        return grad * dy, np.square(x)*dy
    return z, grad_fn


# Python function with custom numerical gradient
@tf.custom_gradient
def foo_custom_grad_numeric(x, y):
    
    epsilon = 0.01
    
    def func(x, y):
        z = np.square(x)*y
        return z
    
    def grad_fn(dy):
        
        # Grad w.r.t x
        J_plus = func(x + epsilon, y)
        J_minus = func(x - epsilon, y)
        gradx = (J_plus -  J_minus)/(2.0*epsilon) 
        
        # Grad w.r.t y
        J_plus = func(x, epsilon + y)
        J_minus = func(x,  - epsilon + y)
        grady = (J_plus -  J_minus)/(2.0*epsilon) 
        
        return gradx * dy, grady*dy
    
    return func(x, y), grad_fn

# Python function only using tf ops with gradient automatically defined via auto-diff 
def foo_autodiff(x, y):
    y = y*tf.square(x)
    return y

with tf.GradientTape(persistent=True) as tape:
    x = tf.constant(2., dtype=tf.float64)
    y = tf.constant(3., dtype=tf.float64)
    tape.watch(x)
    tape.watch(y)
    
    z1 = foo_autodiff(x, y)**2
    z2 = foo_no_grad(x,y)**2
    z3 = foo_custom_grad_analytic(x, y)**2
    z4 = foo_custom_grad_numeric(x, y)**2

print('\nGrad w.r.t. x')
print("foo_autodiff", tape.gradient(z1, x))   
print("foo_no_grad", tape.gradient(z2, x)) 
print("foo_custom_grad_analytic", tape.gradient(z3, x))   
print("foo_custom_grad_numeric", tape.gradient(z4, x))   

print('\nGrad w.r.t. y')
print("foo_no_grad", tape.gradient(z1, y))   
print("foo_custom_grad_analytic", tape.gradient(z2, y))   
print("foo_autodiff", tape.gradient(z3, y))   
print("foo_custom_grad_numeric", tape.gradient(z4, y))  