#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import time
import multiprocessing as mp

from deepafx import lv2_plugin

kLogGradient = False
if kLogGradient:
    kGradient = [] 
def get_param_in_range(params, param_min, param_max):
        """Return the input params (0,1) scaled to (min, max) values."""
        return (param_max-param_min)*params + param_min   
    
    
def rademacher(shape):
    """Generates random samples from a Rademacher distribution +-1
    
    """
    x = np.random.binomial(1, .5, shape)
    x[x==0] = -1
    return x

def uniform(shape):
    return np.random.uniform(-1,1,shape)

def forward(signal, params, dafx, param_map, param_min, param_max, stereo, greedy_pretraining=0):
    """Function applied to each signal of the batch as a function of params."""
        
    # Iterate over the params vector and map it to the correct control on the dafx  
    if isinstance(dafx, lv2_plugin.LV2_Plugin):
        params = get_param_in_range(params, param_min, param_max)
        for i in range(len(params)):
            dafx.set_param(param_map[i], params[i])

        # Process the audio and return
        if stereo:
            return dafx.runs_stereo(np.array(signal).transpose()).astype(np.float32).transpose()
        else:
            return dafx.runs(np.array(signal).transpose()).astype(np.float32).transpose()
    
    elif isinstance(dafx, lv2_plugin.LV2_Plugin_Chain):
        
        if greedy_pretraining == 0:
            greedy_pretraining = len(param_map)
        idx = 0
        out_ = signal.copy()
        
        for j, param_map_plugin in enumerate(param_map[:greedy_pretraining]):
            params_plugin = params[idx:len(param_map_plugin)+idx]
            idx+=len(param_map_plugin)
            dafx_plugin = dafx.plugins[j]
            params_plugin = get_param_in_range(np.array(params_plugin),
                                               np.array(param_min[j]),
                                               np.array(param_max[j]))

            for i in range(len(params_plugin)):
                dafx_plugin.set_param(param_map_plugin[i], params_plugin[i])
            
             # Process the audio
            if stereo[j]:
                out_ = dafx_plugin.runs_stereo(np.array(out_).transpose()).astype(np.float32).transpose()
            else:
                out_ = dafx_plugin.runs(np.array(out_).transpose()).astype(np.float32).transpose()

        out = out_.copy() 
        
        
        return out

def forward_batch(x, y, dafx, param_map, param_min, param_max, stereo):
    """Forward function for an entire batch."""

    # Iterate over batch item
    z = np.zeros_like(x)
    for i in range(x.shape[0]):
        z[i] = forward(x[i], y[i], dafx, param_map, param_min, param_max, stereo)
    return z


def grad_batch_item(dye, xe, ye, dafx_e_plus, dafx_e_minus,
                    param_map, param_min, param_max, epsilon, use_fd, compute_x_grad, stereo,
                    greedy_pretraining=0):
    """Gradient applied to each element of the batch.
        epsilon: delta value for x_grad and SPSA
        epsilons: dictionary """
    
#      epsi
                
    # Grad w.r.t x
    if compute_x_grad:
        c_k = epsilon
        delta_k = rademacher(xe.shape) # parameter twiddle
        J_plus = forward(xe + c_k*delta_k, ye, dafx, param_map, param_min, param_max, stereo, greedy_pretraining=greedy_pretraining)
        J_minus = forward(xe - c_k*delta_k, ye, dafx, param_map, param_min, param_max, stereo, greedy_pretraining=greedy_pretraining)
        gradx = (J_plus - J_minus) / (2 * c_k * delta_k)
        vecJxe = gradx * dye
                        
    else:
        # Return incorrect gradient, assuming this is connected to the input
        vecJxe = np.ones_like(xe)

    # Grad w.r.t y

    # pre-allocate vector * Jaccobian output
#     vecJye = np.empty_like(ye)
    vecJye = np.zeros_like(ye)
                
    if not use_fd: # use SPSA for the signal
        c_k = epsilon
        delta_k = rademacher(ye.shape)
#         delta_k = uniform(ye.shape)# Uncomment to explore uniform distribution
        J_plus = forward(xe, np.clip(ye + c_k*delta_k, 0.0, 1.0), dafx_e_plus,
                         param_map, param_min, param_max,
                         stereo, greedy_pretraining=greedy_pretraining)
        J_minus = forward(xe, np.clip(ye - c_k*delta_k, 0.0, 1.0), dafx_e_minus,
                          param_map, param_min, param_max,
                          stereo, greedy_pretraining=greedy_pretraining)
        grady_num = (J_plus - J_minus) #/ (2 * c_k * delta_k)

    # Iterate over each parameter and compute the output
    
    if greedy_pretraining > 0: 
        limit_parameters = []
        for i in param_map:
            limit_parameters.append(len(i))

        limit_parameters = np.sum(limit_parameters[:greedy_pretraining])
    
    else:
        limit_parameters = ye.shape[0]
        
    for i in range(limit_parameters):
                   
        if use_fd:
            # Forward step
            ye[i] = np.clip(ye[i] + epsilon, 0.0, 1.0)
            J_plus = forward(xe, ye, dafx_e_plus,
                             param_map, param_min, param_max,
                             stereo, greedy_pretraining=greedy_pretraining)

            # Backward step
            ye[i] = np.clip(ye[i] - 2*epsilon, 0.0, 1.0)
            J_minus = forward(xe, ye, dafx_e_minus,
                              param_map, param_min, param_max,
                              stereo, greedy_pretraining=greedy_pretraining)


            # Difference
            grady = (J_plus -  J_minus)/(2.0*epsilon) 
            ye[i] = ye[i] + 1*epsilon

            # TODO: possible do this op all at once
            vecJye[i] = np.dot(np.transpose(dye), grady)
        else:
            grady = grady_num / (2 * c_k * delta_k[i])
            vecJye[i] = np.dot(np.transpose(dye), grady)
            
    
    return vecJxe, vecJye



class Parallel_Batch:
    
    def __init__(self, 
                 multiprocess,
                 num_processes, 
                 plugin_uri, 
                 sr, 
                 param_map, 
                 epsilon, 
                 use_fd, 
                 compute_x_grad,
                 stereo,
                 hop_samples=64,
                 new_params_range=None,
                 non_learnable_params_settings={},
                 fx_chain=False,
                 greedy_pretraining=0,
                 verbose=False):
        
        self.multiprocess = multiprocess
        if not multiprocess:
            num_processes = 1
            
        self.plugin_uri = plugin_uri
        self.sr = sr
        self.hop_samples = hop_samples
        self.verbose = verbose
        self.num_processes = num_processes
        self.param_map = param_map
        self.epsilon = epsilon
        self.stereo = stereo
        self.fx_chain = fx_chain
        self.greedy_pretraining = greedy_pretraining
        self.non_learnable_params_settings = non_learnable_params_settings
        
        self.use_fd = use_fd
        self.compute_x_grad = compute_x_grad
        
        self.plugins = {}
        self.procs = {}
        
        # Create a plugin for each process
        self.plugins = {}
        self.plugins_e_plus = {}
        self.plugins_e_minus = {}
        
        
        for i in range(self.num_processes):
            
            if self.fx_chain:
                
                self.plugins[i] = lv2_plugin.LV2_Plugin_Chain(self.plugin_uri, 
                                                 self.stereo,
                                                 self.sr, 
                                                 hop_samples=self.hop_samples)
                self.plugins_e_plus[i] = lv2_plugin.LV2_Plugin_Chain(self.plugin_uri, 
                                                 self.stereo,
                                                 self.sr, 
                                                 hop_samples=self.hop_samples)
                self.plugins_e_minus[i] = lv2_plugin.LV2_Plugin_Chain(self.plugin_uri, 
                                                 self.stereo,
                                                 self.sr, 
                                                 hop_samples=self.hop_samples)
                
                if self.non_learnable_params_settings:
                    for j, set_nontrainable_parameters_plugin in enumerate(self.non_learnable_params_settings):
                        for p in set_nontrainable_parameters_plugin:
                            self.plugins[i].set_param(j, p, set_nontrainable_parameters_plugin[p])
                            self.plugins_e_plus[i].set_param(j, p, set_nontrainable_parameters_plugin[p])
                            self.plugins_e_minus[i].set_param(j, p, set_nontrainable_parameters_plugin[p])

                noise = 0.01*np.random.normal(0, 1, size=self.hop_samples*10)

                out = self.plugins[i].runs(np.expand_dims(noise,0), greedy_pretraining=self.greedy_pretraining)
                out = self.plugins_e_plus[i].runs(np.expand_dims(noise,0), greedy_pretraining=self.greedy_pretraining)
                out = self.plugins_e_minus[i].runs(np.expand_dims(noise,0), greedy_pretraining=self.greedy_pretraining)
                
            else:
                
                self.plugins[i] = lv2_plugin.LV2_Plugin(self.plugin_uri, 
                                                 self.sr, 
                                                 hop_samples=self.hop_samples)
                self.plugins_e_plus[i] = lv2_plugin.LV2_Plugin(self.plugin_uri, 
                                                 self.sr, 
                                                 hop_samples=self.hop_samples)
                self.plugins_e_minus[i] = lv2_plugin.LV2_Plugin(self.plugin_uri, 
                                                 self.sr, 
                                                 hop_samples=self.hop_samples)

                noise = 0.1*np.random.normal(0, 1, size=self.hop_samples*10)

                if self.stereo:
                    out = self.plugins[i].runs_stereo(np.expand_dims(noise,0))
                    out = self.plugins_e_plus[i].runs_stereo(np.expand_dims(noise,0))
                    out = self.plugins_e_minus[i].runs_stereo(np.expand_dims(noise,0))
                else:
                    out = self.plugins[i].runs(np.expand_dims(noise,0))
                    out = self.plugins_e_plus[i].runs(np.expand_dims(noise,0))
                    out = self.plugins_e_minus[i].runs(np.expand_dims(noise,0))
                    
        self.param_min = []
        self.param_max = []
        self.default_values = []
        if self.fx_chain is False:            
            self.param_range = {}
            for k in self.param_map:
                d, param_min_, param_max_ = self.get_param_range(self.param_map[k])
                self.default_values.append(float(str(d)))
                self.param_range[k] = [float(str(param_min_)), float(str(param_max_))]
                
        elif self.fx_chain:  

            self.param_range = []           
            for i, param_map_plugin in enumerate(self.param_map): 
                param_range_dict = {}
                default_values = []
                for k in param_map_plugin:
                    d, param_min_, param_max_ = self.get_param_range(param_map_plugin[k], plugin_id=i)
                    default_values.append(float(str(d)))
                    param_range_dict[k] = [float(str(param_min_)), float(str(param_max_))]
                self.default_values.append(default_values)
                self.param_range.append(param_range_dict)

        self.update_param_min_max()
        
        if self.fx_chain is False:      
            self.new_params_range = new_params_range
            if self.new_params_range:
                self.modify_parameter_range(self.new_params_range)
                
        elif self.fx_chain:   
            for i, new_params_range_plugin in enumerate(new_params_range): 
                if new_params_range_plugin:
                    self.modify_parameter_range(new_params_range_plugin, plugin_id=i)
   
            
    def modify_parameter_range(self, new_param_range, plugin_id=None):
        
        # not pretty, but translates id from plugin to id in param map and then updates the param range.
        
        if self.fx_chain: 
            
            new_param_range_ = {}
            for i in new_param_range:
                key = list(self.param_map[plugin_id].keys())[list(self.param_map[plugin_id].values()).index(i)]
                new_param_range_[key] = list(new_param_range[self.param_map[plugin_id][key]])

            for i in new_param_range_:
                self.param_range[plugin_id][i] = new_param_range_[i]
            self.update_param_min_max()
            
        else:
            
            new_param_range_ = {}
            for i in new_param_range:
                key = list(self.param_map.keys())[list(self.param_map.values()).index(i)]
                new_param_range_[key] = list(new_param_range[self.param_map[key]])

            for i in new_param_range_:
                self.param_range[i] = new_param_range_[i]
            self.update_param_min_max()
            
    def update_param_min_max(self):
        
        self.param_min = []
        self.param_max = []
        
        if self.fx_chain: 
            
            for param_range_plugin in self.param_range:
                param_min_list = []
                param_max_list = []
                for k in param_range_plugin:
                    param_min_list.append(param_range_plugin[k][0])    
                    param_max_list.append(param_range_plugin[k][1])
                self.param_min.append(param_min_list)    
                self.param_max.append(param_max_list)
      
        else:
            
            for k in self.param_range:
                self.param_min.append(self.param_range[k][0])    
                self.param_max.append(self.param_range[k][1])

            self.param_min = np.asarray(self.param_min)
            self.param_max = np.asarray(self.param_max)
        
    def init(self):
    
        if self.multiprocess:
            procs = {}
            for i in range(self.num_processes):

                plugin = self.plugins[i]
                plugin_e_plus = self.plugins_e_plus[i]
                plugin_e_minus = self.plugins_e_minus[i]

                # create a pipe between the worker and main thread 

                parent_conn, child_conn = mp.Pipe()
                p = mp.Process(target=Parallel_Batch.worker_pipe, 
                               args=(parent_conn, 
                                     child_conn, 
                                     plugin,
                                     plugin_e_plus,
                                     plugin_e_minus,
                                     self.param_map,
                                     self.param_min,
                                     self.param_max,
                                     self.epsilon,
                                     self.use_fd, 
                                     self.compute_x_grad,
                                     self.stereo,
                                     self.greedy_pretraining)) 

                # running processes 
                p.start() 

                # Store the process, pipes/queues
                procs[i] = [p, parent_conn, child_conn]
            
            self.procs = procs
    
    def __del__(self):
        self.shutdown()

    def worker_pipe(parent_conn, 
                    child_conn,
                    dafx,
                    dafx_e_plus,
                    dafx_e_minus,
                    param_map,
                    param_min,
                    param_max,
                    epsilon,
                    use_fd, 
                    compute_x_grad,
                    stereo,
                    greedy_pretraining): 
        """ 
        Worker function to be executed on a separate process
        """

        while True: 

            # Read messages from the pipe
            msg, value = child_conn.recv() 

            # Try to interpret an END kill message
            try:
                if str(msg) == 'end': 
                    break
                elif str(msg) == 'grad':
                    dye, xe, ye = value
                    vecJxe, vecJye = grad_batch_item(dye, xe, ye, 
                                                     dafx_e_plus,
                                                     dafx_e_minus,
                                                     param_map,
                                                     param_min,
                                                     param_max,
                                                     epsilon,
                                                     use_fd, 
                                                     compute_x_grad,
                                                     stereo,
                                                     greedy_pretraining=greedy_pretraining)

                    # Return the output signal
                    child_conn.send((vecJxe, vecJye))
                elif str(msg) == 'forward':
                    signal, params = value
                    output = forward(signal,
                                     params,
                                     dafx,
                                     param_map,
                                     param_min,
                                     param_max,
                                     stereo,
                                     greedy_pretraining=greedy_pretraining)
                    child_conn.send(output)
                    
                elif str(msg) == 'set_param':
                    
                    param_id, v = value
                    dafx.set_param(param_id, v)
                    dafx_e_plus.set_param(param_id, v)
                    dafx_e_minus.set_param(param_id, v)
                                
                elif str(msg) == 'reset':
                    
                    samples = value
                    noise = 0.01*np.random.normal(0, 1, size=samples)
                    noise = np.expand_dims(noise,0)
                    
                    if self.fx_chain: 
                        
                        out = dafx.runs(noise, greedy_pretraining=greedy_pretraining)
                        out = dafx_e_plus.runs(noise, greedy_pretraining=greedy_pretraining)
                        out = dafx_e_minus.runs(noise, greedy_pretraining=greedy_pretraining)
                            
                    else:
 
                        if stereo:
                            out = dafx.runs_stereo(noise)
                            out = dafx_e_plus.runs_stereo(noise)
                            out = dafx_e_minus.runs_stereo(noise)
                        else:
                            out = dafx.runs(noise)
                            out = dafx_e_plus.runs(noise)
                            out = dafx_e_minus.runs(noise)
                    
            except:
                pass
            
    def reset_plugin_state(self, samples):
        
        if self.multiprocess:
            for i in self.plugins:
                msg = ('reset', (samples))
                self.procs[i][1].send(msg)     
                
        else:
            
            noise = 0.01*np.random.normal(0, 1, size=samples)
            noise = np.expand_dims(noise,0)
            
            for i in self.plugins:
            
                if self.fx_chain: 

                    out = self.plugins[i].runs(noise, greedy_pretraining=self.greedy_pretraining)
                    out = self.plugins_e_plus[i].runs(noise, greedy_pretraining=self.greedy_pretraining)
                    out = self.plugins_e_minus[i].runs(noise, greedy_pretraining=self.greedy_pretraining)

                else:

                    if self.stereo:
                        out = self.plugins[i].runs_stereo(noise)
                        out = self.plugins_e_plus[i].runs_stereo(noise)
                        out = self.plugins_e_minus[i].runs_stereo(noise)
                    else:
                        out = self.plugins[i].runs(noise)
                        out = self.plugins_e_plus[i].runs(noise)
                        out = self.plugins_e_minus[i].runs(noise)

        
    def shutdown(self):
        
        # Send the final kill/END message
        for key in self.procs:
            self.procs[key][1].send(("end", 0))
            
        # Join: Wait until processes finish (kills the thread)
        for key in self.procs:
            self.procs[key][0].join() 
            
        # Clear all the plugins
        self.plugins = {}    
        self.procs = {}
 
    
    def run_grad_batch(self, dy, x, y):
        
        if self.multiprocess:
            # Loop over the audio batch to launch the processing
            for i in range(dy.shape[0]):
                msg = ('grad', (dy[i], x[i], y[i]))
                self.procs[i][1].send(msg) 

            # Wait for the return message from each process 
            vecJx = np.empty_like(x)
            vecJy = np.empty_like(y)
            for i in range(dy.shape[0]):
                vecJx[i], vecJy[i] = self.procs[i][1].recv()
            if kLogGradient:
                kGradient.append(vecJy)
            return vecJx, vecJy
        
        else:
            
            # Single process/threaded
            vecJx = np.empty_like(x)
            vecJy = np.empty_like(y)
            for i in range(dy.shape[0]):
                vecJx[i], vecJy[i] = grad_batch_item(dy[i], x[i], y[i], 
                                                 self.plugins_e_plus[0],
                                                 self.plugins_e_minus[0],
                                                 self.param_map, 
                                                 self.param_min,
                                                 self.param_max,
                                                 self.epsilon,
                                                 self.use_fd, 
                                                 self.compute_x_grad,
                                                    self.stereo)
            return vecJx, vecJy
    
    def get_param_range(self, index, plugin_id=0):
        
        if isinstance(self.plugin_uri, list):
            param_range = self.plugins[0].get_param_range(plugin_id, index) 
            
        else:
            param_range = self.plugins[0].get_param_range(index) 
        
        return param_range

    def set_param(self, param_id, value, plugin_id=None):
        """Set a parameter value given the parameter id"""
        
        if self.fx_chain is False: 
            
            if self.multiprocess:
                for i in self.plugins:
                    msg = ('set_param', (param_id, value))
                    self.procs[i][1].send(msg)     

            else:
                for i in self.plugins:
                    plugin = self.plugins[i]
                    plugin.set_param(param_id, value)
            

        # TO DO: The following code doesn't work because msg has three parameters
#         else:
            
#             if self.multiprocess:
#                 for i in self.plugins:
#                     msg = ('set_param_fx_chain', (param_id, value, plugin_id))
#                     self.procs[i][1].send(msg)     

#             else:
#                 for i in self.plugins:
#                     plugin = self.plugins[i]
#                     plugin.set_param(plugin_id, param_id, value)
            
            
    
    def run_forward_batch(self, x, y):
        
        
        if self.multiprocess:
            
            # Loop over the audio batch to launch the processing
            for i in range(x.shape[0]):
                msg = ('forward', (x[i], y[i]))
                self.procs[i][1].send(msg) 

            # Wait for the return message from each process 
            z = np.empty_like(x)
            for i in range(x.shape[0]):
                z[i] = self.procs[i][1].recv()
            return z
        
        else:
            return forward_batch(x, y, self.plugins[0], self.param_map, self.param_min, self.param_max, self.stereo)

    
def setup_custom_dafx_op(mb):
    
    """Returns a LV2 custom TF operator with custom gradient.
       
       param_map - map from index (0-N) to parameter id on the LV2 plugin
       gradient_method : str
           fd (finite differences) or spsa (simultaneous permutations stochastic approximation)
    
    """

    # initialize batch operator
    # TODO: figure out why this needs to happen here
    mb.init()
        
    @tf.custom_gradient
    def custom_grad_numeric_batch(x, y):
        """Custom TF operator with custom numerical gradient.
        
        x - tf.tensor
            batchsize x signal_length x 1
        y - tf.tensor
            batchsize x num_params
        """

        # Convert from TF tensors to np arrays
        x = np.array(x)
        y = np.array(y)

        def grad_batch(dy):
            """Gradient applied to each batch."""
            dy = np.array(dy) # speed up
            return mb.run_grad_batch(dy, x, y)

        # Call the forward function
        z = mb.run_forward_batch(x, y)

        return z, grad_batch

    return custom_grad_numeric_batch



# Create a Keras Layer for the black-box audio FX
class DAFXLayer(tf.keras.layers.Layer):
    """A differentiable LV2 DAFX Keras Layer. 
    
    This Keras layer uses numerical differentation over specified parameters 
    of an LV2 DAFX plugin. You can specific what parameters to control via a
    deep network, what parameters to hold fixed, and call it like any other 
    Keras layer. Currently assumes the first input is a non-learnable signal 
    and the second input is a learnable set of parameters.
    
    """
    
    def __init__(self, 
                 plugin_uri, 
                 sr, 
                 hop_samples, 
                 param_map, 
                 non_learnable_params_settings={},
                 new_params_range={},
                 gradient_method='fd',
                 compute_signal_grad=False,
                 multiprocess=False,
                 num_multiprocess=32,
                 stereo=False,
                 fx_chain=False,
                 greedy_pretraining=0,
                 **kwargs):
        """
        Parameters
        ----------
        plugin_uri : str
            The LV2 plugin uri
        sr : int
            Sampling rate of the plugin
        hop_samples : int
            Plugin host render hop size in samples.
        param_map : dict int:int
            Dictionary of flat vector index to id on the DAFX
        gradient_method : str
            String specifying the gradient method used for backprop: fd or spsa
        """
    
        super(DAFXLayer, self).__init__(**kwargs)

        self.plugin_uri = plugin_uri
        self.sr = sr
        self.hop_samples = hop_samples
        self.gradient_method = gradient_method
        self.compute_signal_grad = compute_signal_grad
        self.multiprocess = multiprocess
        self.num_multiprocess = num_multiprocess
        self.stereo = stereo
        self.fx_chain = fx_chain
        self.greedy_pretraining = greedy_pretraining
        
        # Enforce param map is int to int
        if self.fx_chain is False:
            self.param_map = {}
            for k in param_map:
                self.param_map[int(k)] = int(param_map[k]) 

        else:
            self.param_map = []
            for param_map_plugin in param_map:
                param_map_dict = {}
                for k in param_map_plugin:
                    param_map_dict[int(k)] = int(param_map_plugin[k]) 
                self.param_map.append(param_map_dict)
        
        if self.fx_chain is False:
            self.non_learnable_params_settings = {}
            for k in non_learnable_params_settings:
                index = int(k)
                value = non_learnable_params_settings[k]
                self.non_learnable_params_settings[index] = value
                
        else:
            self.non_learnable_params_settings = []
            for non_learnable_params_settings_plugin in non_learnable_params_settings:
                non_learnable_params_settings_dict = {}
                for k in non_learnable_params_settings_plugin:
                    index = int(k)
                    value = non_learnable_params_settings_plugin[k]
                    non_learnable_params_settings_dict[index] = value
                self.non_learnable_params_settings.append(non_learnable_params_settings_dict)
               
        self.epsilon = .01 # epsilon for signal, FD and SPSA
        self.use_fd = gradient_method == 'fd'
        self.compute_x_grad = compute_signal_grad
        self.new_params_range = new_params_range
        self.mb = Parallel_Batch(self.multiprocess,
                                 self.num_multiprocess, 
                                 self.plugin_uri, 
                                 self.sr, 
                                 self.param_map, 
                                 self.epsilon, 
                                 self.use_fd, 
                                 self.compute_x_grad,
                                 self.stereo,
                                 hop_samples=self.hop_samples,
                                 new_params_range=self.new_params_range,
                                 fx_chain=self.fx_chain,
                                 non_learnable_params_settings = self.non_learnable_params_settings,
                                 greedy_pretraining=self.greedy_pretraining
                                 )
               

    def shutdown(self):
        self.mb.shutdown()
            
    def __del__(self):
        self.mb.shutdown()
        del self.mb
        
    def reset_dafx_state(self, samples):
        samples = (samples//self.hop_samples)*self.hop_samples
        self.mb.reset_plugin_state(samples)
        
    def set_greedy_pretraining(self, greedy_pretraining):
        
        self.greedy_pretraining = greedy_pretraining
        
        self.mb.shutdown()
        self.mb = Parallel_Batch(self.multiprocess,
                                 self.num_multiprocess, 
                                 self.plugin_uri, 
                                 self.sr, 
                                 self.param_map, 
                                 self.epsilon, 
                                 self.use_fd, 
                                 self.compute_x_grad,
                                 self.stereo,
                                 hop_samples=self.hop_samples,
                                 new_params_range=self.new_params_range,
                                 fx_chain=self.fx_chain,
                                 non_learnable_params_settings = self.non_learnable_params_settings,
                                 greedy_pretraining=self.greedy_pretraining
                                 )
        # Create the DAFX op and LV2 plugins
        self.dafx_func = setup_custom_dafx_op(self.mb)


    def set_dafx_param(self, index, value, plugin_id=0):
        """Set fixed parameter values and applied at build time.
        
        Parameters
        ----------
        index : int
            index into the vector of estimated parameters
        value : int
            parameter id of the DAFX associated with the estimated parameter
        """
        
        # Store a version of the set parameters to serialize
        # TO DO. check why unexpected behavoir when fx_chain is false 
        
        if self.fx_chain:
            self.non_learnable_params_settings[plugin_id][int(index)] = int(value)
        else:
            self.non_learnable_params_settings[int(index)] = int(value)
       
 
        
    def get_dafx_param_range(self, index, plugin_id=0):
        """Return the port (default, min, max) values."""
        if self.fx_chain:
            return self.mb.get_param_range(index, plugin_id=pluding_id)
        else:
            return self.mb.get_param_range(index)
    

    def build(self, input_shape):
        """Build the layer based on the input shape."""
        
        super(DAFXLayer, self).build(input_shape)

        # Create the DAFX op and LV2 plugins
        self.dafx_func = setup_custom_dafx_op(self.mb)
        
       
        
        # Set the parameters before building
        if self.fx_chain is False:
            for i in self.non_learnable_params_settings:
                self.mb.set_param(i, self.non_learnable_params_settings[i])
                
##    TODO, mb, multiprocess is not working with msg set_param_fx_chain
#         else:
#             for j, non_learnable_params_settings_plugin in enumerate(self.non_learnable_params_settings):
#                 for i in non_learnable_params_settings_plugin:
#                     self.mb.set_param(j, i, non_learnable_params_settings_plugin[i])


    def call(self, inputs, training=None):
        """Invoke the layer processing."""
        
        # TODO: disable grad computation in py_function if not training
        x = inputs[0]
        params = inputs[1]
        ret = tf.py_function(func=self.dafx_func, 
                             inp=[x, params], 
                             Tout=tf.float32)
        ret.set_shape(x.get_shape())

        return ret
    
    
    def get_config(self):
        """Serialize the layer state needed for saving/reloading."""

        config = {
            'plugin_uri': self.plugin_uri,
            'sr': self.sr,
            'hop_samples': self.hop_samples,
            'param_map': self.param_map,
            'non_learnable_params_settings': self.non_learnable_params_settings,
            'gradient_method': self.gradient_method,
            'compute_signal_grad': self.compute_signal_grad, 
            'num_multiprocess': self.num_multiprocess
        }
        base_config = super(DAFXLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


