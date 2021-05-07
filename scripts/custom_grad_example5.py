#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/

# Custom gradient example with a batch of vectors (signal) and vectors (of parameter) arguments
# f(x, y) = x^2 * prod(y), where x and y are a batch of vectors
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


@tf.custom_gradient
def foo_custom_grad_analytic_batch(x, y):
    """Custom analytic gradient for a batch of vectors."""
    
    def _func(xe, ye):
        """Function applied to each element of the batch."""
        ze = np.square(xe)*np.prod(ye)
        return ze
    
    # Iterate over batch item
    z = []
    for i in range(x.shape[0]):
        z.append(_func(x[i], y[i]))
    z = tf.stack(z)
    
    def grad_fn(dy):
        """Gradient applied to each element of the batch."""
        def _grad_fn(dye, xe, ye):
        
            # Gradient w.r.t x
            gradx = 2 * np.array(xe) * np.prod(ye)

            # Gradient w.r.t y
            vecJye = np.zeros_like(ye)
            for i in range(ye.shape[0]):
                vecJye[i] = np.prod(np.delete(ye, [i]))*np.dot(np.transpose(dye), np.square(xe))

            vecJxe = gradx * dye
            # Returns vector-Jacobian
            return vecJxe, vecJye
        
        dy1 = []
        dy2 = []
        for i in range(dy.shape[0]):
            vecJxe, vecJye = _grad_fn(dy[i], x[i], y[i])
            dy1.append(vecJxe)
            dy2.append(vecJye)
        return tf.stack(dy1), tf.stack(dy2)

    return z, grad_fn


@tf.custom_gradient
def foo_custom_grad_numeric_batch(x, y):
    """Custom numeric gradient for a batch of vectors."""
    epsilon = 0.001
    
    def _func(xe, ye):
        """Function applied to each element of the batch."""
        ze = np.square(xe)*np.prod(ye)
        return ze
    
    # Iterate over batch item
    z = []
    for i in range(x.shape[0]):
        z.append(_func(x[i], y[i]))
    z = tf.stack(z)
    
    def grad_fn(dy):
        """Gradient applied to each element of the batch."""

        def _grad_fn(dye, xe, ye):

            # Grad w.r.t x
            J_plus = _func(xe + epsilon, ye)
            J_minus = _func(xe - epsilon, ye)
            gradx = (J_plus -  J_minus)/(2.0*epsilon) 
            vecJxe = gradx * dye

            # Grad w.r.t y
            yc = ye.numpy()

            # pre-allocate vector * Jaccobian output
            vecJye = np.zeros_like(ye)

            # Iterate over each parameter and compute the output
            for i in range(ye.shape[0]):

                yc[i] = yc[i] + epsilon
                J_plus = _func(xe, yc)
                yc[i] = yc[i] - 2*epsilon
                J_minus = _func(xe, yc)
                grady = (J_plus -  J_minus)/(2.0*epsilon) 
                yc[i] = yc[i] + 1*epsilon
                vecJye[i] = np.dot(np.transpose(dye), grady)

            return vecJxe, vecJye
    
        
        dy1 = []
        dy2 = []
        for i in range(dy.shape[0]):
            vecJxe, vecJye = _grad_fn(dy[i], x[i], y[i])
            dy1.append(vecJxe)
            dy2.append(vecJye)
        return tf.stack(dy1), tf.stack(dy2)

    return z, grad_fn


# Python function with custom numerical gradient
@tf.custom_gradient
def foo_custom_grad_numeric(x, y):
    """Custom numeric gradient for vectors."""
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
    """Auto-diff gradient for a batch of vectors."""
    
    def _foo_autodiff(x,y):
        z = tf.math.reduce_prod(y)*tf.square(x)
        return z
        
    # Iterate over batch item
    transformed_images = []
    for i in range(x.shape[0]):
        transformed_images.append(_foo_autodiff(x[i], y[i]))
    result = tf.stack(transformed_images)
    return result

with tf.GradientTape(persistent=True) as tape:
    
    a = 2*np.ones((2,5,1)); a[0,0,0] = 1# signal, shape = batch x time x 1
    b = 3*np.ones((2,3,1)) #b[0,1:2,0] = 1 # params, shape = batch x params x 1
    x = tf.constant(a, dtype=tf.float64)
    y = tf.constant(b, dtype=tf.float64)
    tape.watch(x)
    tape.watch(y)
    
    z3 = foo_autodiff(x, y)**2
    z1 = foo_no_grad(x,y)**2
    z2 = foo_custom_grad_analytic_batch(x, y)**2
    z4 = foo_custom_grad_numeric_batch(x, y)**2


print('\nGrad w.r.t. x')
print("foo_autodiff", tape.gradient(z1, x))   
print("foo_no_grad", tape.gradient(z2, x)) 
print("foo_custom_grad_analytic", tape.gradient(z3, x))   
print("foo_custom_grad_numeric", tape.gradient(z4, x))  

print('\nGrad w.r.t. y')
print("foo_autodiff", tape.gradient(z1, y))  
print("foo_no_grad", tape.gradient(z2, y))  
print("foo_custom_grad_analytic", tape.gradient(z3, y))   
print("foo_custom_grad_numeric", tape.gradient(z4, y))   