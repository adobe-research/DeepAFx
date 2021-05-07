#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/

# Custom gradient example with one scalar argument of the function:
# f(x) = x^2
import numpy as np
import tensorflow as tf


# Python function with no custom gradient
def foo_no_grad(x):
    y = np.square(x)
    return tf.constant(y)


# Python function with custom analytic gradient
@tf.custom_gradient
def foo_custom_grad_analytic(x):
    y = np.square(x)
    def grad_fn(dy):
        grad = 2 * np.array(x)
        return grad * dy
    return y, grad_fn


# Python function with custom numerical gradient
@tf.custom_gradient
def foo_custom_grad_numeric(x):
    
    epsilon = 0.01
    
    def func(x):
        y = np.square(x)
        return y
    
    def grad_fn(dy):
        
        J_plus = func(x + epsilon)
        J_minus = func(x - epsilon)
        grad = (J_plus -  J_minus)/(2.0*epsilon) 
        return grad * dy
    
    return func(x), grad_fn

# Python function only using tf ops with gradient automatically defined via auto-diff 
def foo_autodiff(x):
    y = tf.square(x)
    return y

with tf.GradientTape(persistent=True) as tape:
    x = tf.constant(2., dtype=tf.float64)
    tape.watch(x)
    
    y1 = foo_autodiff(x)**2
    y2 = foo_no_grad(x)**2
    y3 = foo_custom_grad_analytic(x)**2
    y4 = foo_custom_grad_numeric(x)**2

print("Auto-diff:", tape.gradient(y1, x))  
print("No-diff:", tape.gradient(y2, x))   
print("Analytic gradient:", tape.gradient(y3, x))   
print("Numerical gradient:", tape.gradient(y4, x))  



