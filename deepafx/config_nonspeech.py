#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
import numpy as np

k = {}

k['sr'] = 22050
k['num_samples'] = 40960
k['batch_size'] = 100
k['steps_per_epoch'] = 1000
k['epochs'] = 1000
k['patience'] = 25
k['encoder'] = 'mobilenet' # inception or mobilenet

# Define paths
k['path_audio'] = '/home/code-base/scratch_space/nonspeech/daps_22050/'
k['x_recording'] = 'cleanraw/'
k['y_recording'] = 'clean/'
k['path_models'] = '/home/code-base/scratch_space/models/nonspeech/'


# DAFX constants
k['output_length'] = 1024 # Length of output frame
k['hop_samples'] = 1024 # Length of hop size, for non-overlapping frames it should be equal to output_length
k['gradient_method'] = 'spsa' # or 'spsa'
k['compute_signal_gradient'] = False
k['multiprocess'] = True
k['greedy_dafx_pretraining'] = True # Enables progressive training
k['default_pretraining'] = True # Enables default initialization of parameter values for training

# Single DAFx:

# Multiband compressor

# k['plugin_uri'] = 'http://calf.sourceforge.net/plugins/MultibandCompressor'

# k['stereo'] = True
# # params = [19,20,30,31,41,42,52,53,15,16,17] # threshold, ratio, freqs.
# # params = [19,20,23,30,31,34,41,42,45,52,53,56,15,16,17] # threshold, ratio, makeupgain, freqs.
# # params = [19,20,23,30,31,34,41,42,45,52,53,56,15,16,17,5] # threshold, ratio, makeupgain, freqs, out gain
# # params = [19,20,23,30,31,34,41,42,45,52,53,56,15,16,17,5,6] # threshold, ratio, makeupgain, freqs, input output gain gain

# params = [19,20,23,24,30,31,34,35,41,42,45,46,52,53,56,57,15,16,17,5,6] # threshold, ratio, makeupgain, knee, freqs, input output  gain

# k['set_nontrainable_parameters'] = {28:0,
#                                     39:0,
#                                     50:0,
#                                     61:0,
#                                     21:0.01,
#                                     22:0.01,
#                                     32:0.01,
#                                     33:0.01,
#                                     43:0.01,
#                                     44:0.01,
#                                     54:0.01,
#                                     55:0.01,
#                                    }

# k['new_parameter_range'] = {15:[10.0, 300.0],
#                             16:[300.0, 3000.0],
#                            17:[3000.0,8000.0],
#                            5:[0.015625,2],
#                            6:[0.015625,2]
#                            }


# Multiband Gate

# k['plugin_uri'] = 'http://calf.sourceforge.net/plugins/MultibandGate'
# k['stereo'] = True

# # params = [19,20,21,31,32,33,43,44,45,55,56,57,15,16,17] # bands +freq split 1/2 2/3 and 3/4
# # params = [19,20,21,31,32,33,43,44,45,55,56,57,15,16] # bands +freq split 1/2 2/3 
# # params = [19,20,21,31,32,33,43,44,45,55,56,57,15,16,17] # bands +freq split 1/2 2/3 3/4
# params = [19,20,21,31,32,33,43,44,45,55,56,57,15,16,17,5,6] # bands +freq split 1/2 2/3 3/4 + input output gain

# k['set_nontrainable_parameters'] = {29:0,
#                                     41:0,
#                                     53:0,
#                                     65:0,
#                                     22:0.01,
#                                     23:0.01,
#                                     34:0.01,
#                                     35:0.01,
#                                     46:0.01,
#                                     47:0.01,
#                                     58:0.01,
#                                     59:0.01,
#                                    }

# k['new_parameter_range'] = {15:[10.0, 300.0],
#                             16:[300.0, 3000.0],
#                            17:[3000.0,10000.0],
#                            5:[0.015625, 2.0],
#                            6:[0.015625,2.0]}


# 32 band graphic EQ mono

# k['plugin_uri'] = 'http://lsp-plug.in/plugins/lv2/graph_equalizer_x32_mono'

# k['stereo'] = False

# params = [18]
# for i in range(31):
#     params.append(params[i]+5)
    
# k['new_parameter_range'] = {}
# for i in params:
#     k['new_parameter_range'][i] = [0.015850, 4.0]
    
# params.append(3)
# params.append(4)

# k['new_parameter_range'][3] = [0.0, 4.0]
# k['new_parameter_range'][4] = [0.0, 4.0]


# k['set_nontrainable_parameters'] = {6:4}


# # 32 band Parametric EQ mono

# k['plugin_uri'] = 'http://lsp-plug.in/plugins/lv2/para_equalizer_x32_mono'

# k['stereo'] = False

# # # trainable parameters: freq and gain
# params = [19,20]

# for i in np.linspace(0,60,31):
#     i = int(i)
#     params.append(params[i]+10)
#     params.append(params[i+1]+10)
    
# # parameter range for all freqs and gains:

# k['new_parameter_range'] = {}

# for i in np.linspace(0,60,31):
#     i = int(i)
#     k['new_parameter_range'][params[i]] = [10.0, 11025.0]
#     k['new_parameter_range'][params[i+1]] = [0.015850, 4.0]

# # parameter range for highpass and lowpass filters:

# k['new_parameter_range'][19] = [10.0, 100.0]
# k['new_parameter_range'][329] = [10000.0, 11025.0]
    

# # set nontrainable parameters: type filter, mode filter, Q

# non_trainable_params = [14,15,21]
# k['set_nontrainable_parameters'] = {}

# for i in np.linspace(0,90,31):
#     i = int(i)
#     non_trainable_params.append(non_trainable_params[i]+10)
#     non_trainable_params.append(non_trainable_params[i+1]+10)
#     non_trainable_params.append(non_trainable_params[i+2]+10)
    
# for i in np.linspace(0,90,31):
#     i = int(i)
#     k['set_nontrainable_parameters'][non_trainable_params[i]] = 1.0 # type = bell
#     k['set_nontrainable_parameters'][non_trainable_params[i+1]] = 2.0 # mode = "BWC (BT)"
#     k['set_nontrainable_parameters'][non_trainable_params[i+2]] = 4.0 # Q = 1
    
# # set nontrainable parameters: type filter, mode filter

# # uncoment when adding Q as trainable parameter

# # non_trainable_params = [14,15,21]
# # k['set_nontrainable_parameters'] = {}

# # for i in np.linspace(0,90,31):
# #     i = int(i)
# #     non_trainable_params.append(non_trainable_params[i]+10)
# #     non_trainable_params.append(non_trainable_params[i+1]+10)
# #     non_trainable_params.append(non_trainable_params[i+2]+10)

# # # adding Q a fix range (find nicer way to do this)
# # for i in np.linspace(0,93,32):
# #     i = int(i)
# #     params.append(non_trainable_params[i+2])
# #     k['new_parameter_range'][non_trainable_params[i+2]] = [0.0, 50.0]
    
# # for i in np.linspace(0,90,31):
# #     i = int(i)
# #     k['set_nontrainable_parameters'][non_trainable_params[i]] = 1.0 # type = bell
# #     k['set_nontrainable_parameters'][non_trainable_params[i+1]] = 2.0 # mode = "BWC (BT)"
# # #     k['set_nontrainable_parameters'][non_trainable_params[i+2]] = 1.0 # Q = 1

# # set nontrainable parameters for highpass, lowshelf, highshelf, lowpass: type filter

# k['set_nontrainable_parameters'][14] = 2 #highpass
# k['set_nontrainable_parameters'][24] = 5 #lowshelf
# k['set_nontrainable_parameters'][314] = 3 #highshelf
# k['set_nontrainable_parameters'][324] = 4 #lowpass

# # Add input_gain and output_gain

# params.append(3)
# params.append(4)

# k['new_parameter_range'][3] = [0.0, 4.0]
# k['new_parameter_range'][4] = [0.0, 4.0]


# k['params'] = len(params)
# k['param_map'] = {}
# for i, port in enumerate(params):
#     k['param_map'][i] = port
    
    
    
# MULTIPLE DAFx:


k['params'] = []
k['plugin_uri'] = []
k['param_map'] = []
k['stereo'] = []
k['set_nontrainable_parameters'] = []
k['new_parameter_range'] = []
    
    
# Multiband Gate

plugin_uri = 'http://calf.sourceforge.net/plugins/MultibandGate'
stereo = True

# params = [19,20,21,31,32,33,43,44,45,55,56,57,15,16,17] # bands +freq split 1/2 2/3 and 3/4
# params = [19,20,21,31,32,33,43,44,45,55,56,57,15,16] # bands +freq split 1/2 2/3 
# params = [19,20,21,31,32,33,43,44,45,55,56,57,15,16,17] # bands +freq split 1/2 2/3 3/4
parameters = [19,20,21,31,32,33,43,44,45,55,56,57,15,16,17,5,6] # bands +freq split 1/2 2/3 3/4 + input output gain

set_nontrainable_parameters = {29:0,
                                    41:0,
                                    53:0,
                                    65:0,
                                    22:0.01,
                                    23:0.01,
                                    34:0.01,
                                    35:0.01,
                                    46:0.01,
                                    47:0.01,
                                    58:0.01,
                                    59:0.01,
                                   }

new_parameter_range = {15:[10.0, 300.0],
                            16:[300.0, 3000.0],
                           17:[3000.0,10000.0],
                           5:[0.015625, 2.0],
                           6:[0.015625,2.0]}

    
params = len(parameters)
param_map = {}
for i, port in enumerate(parameters):
    param_map[i] = port
k['params'].append(params)
k['plugin_uri'].append(plugin_uri)
k['param_map'].append(param_map)
k['stereo'].append(stereo)
k['set_nontrainable_parameters'].append(set_nontrainable_parameters)
k['new_parameter_range'].append(new_parameter_range)

# # 32 Band Graph EQ

# plugin_uri = 'http://lsp-plug.in/plugins/lv2/graph_equalizer_x32_mono'
# stereo = False

# parameters = [18]
# for i in range(31):
#     parameters.append(parameters[i]+5)
    
# new_parameter_range = {}
# for i in parameters:
#     new_parameter_range[i] = [0.015850, 5.0]
    
# parameters.append(3)
# parameters.append(4)

# new_parameter_range[3] = [0.0, 4.0]
# new_parameter_range[4] = [0.0, 4.0]


# set_nontrainable_parameters = {6:4}

# params = len(parameters)
# param_map = {}
# for i, port in enumerate(parameters):
#     param_map[i] = port
    
# k['params'].append(params)
# k['plugin_uri'].append(plugin_uri)
# k['param_map'].append(param_map)
# k['stereo'].append(stereo)
# k['set_nontrainable_parameters'].append(set_nontrainable_parameters)
# k['new_parameter_range'].append(new_parameter_range)






# # 32 band Parametric EQ mono

# plugin_uri = 'http://lsp-plug.in/plugins/lv2/para_equalizer_x32_mono'
# stereo = False

# # # trainable parameters: freq and gain
# parameters = [19,20]

# for i in np.linspace(0,60,31):
#     i = int(i)
#     parameters.append(parameters[i]+10)
#     parameters.append(parameters[i+1]+10)
    
# # parameter range for all freqs and gains:

# new_parameter_range = {}

# for i in np.linspace(0,60,31):
#     i = int(i)
#     new_parameter_range[parameters[i]] = [10.0, 11025.0]
#     new_parameter_range[parameters[i+1]] = [0.015850, 4.0]

# # parameter range for highpass and lowpass filters:

# new_parameter_range[19] = [10.0, 100.0]
# new_parameter_range[329] = [10000.0, 11025.0]
    

# # set nontrainable parameters: type filter, mode filter, Q

# non_trainable_params = [14,15,21]
# set_nontrainable_parameters = {}

# for i in np.linspace(0,90,31):
#     i = int(i)
#     non_trainable_params.append(non_trainable_params[i]+10)
#     non_trainable_params.append(non_trainable_params[i+1]+10)
#     non_trainable_params.append(non_trainable_params[i+2]+10)
    
# for i in np.linspace(0,90,31):
#     i = int(i)
#     set_nontrainable_parameters[non_trainable_params[i]] = 1.0 # type = bell
#     set_nontrainable_parameters[non_trainable_params[i+1]] = 2.0 # mode = "BWC (BT)"
#     set_nontrainable_parameters[non_trainable_params[i+2]] = 4.0 # Q = 1
    
# # set nontrainable parameters: type filter, mode filter

# # uncoment when adding Q as trainable parameter

# # non_trainable_params = [14,15,21]
# # k['set_nontrainable_parameters'] = {}

# # for i in np.linspace(0,90,31):
# #     i = int(i)
# #     non_trainable_params.append(non_trainable_params[i]+10)
# #     non_trainable_params.append(non_trainable_params[i+1]+10)
# #     non_trainable_params.append(non_trainable_params[i+2]+10)

# # # adding Q a fix range (find nicer way to do this)
# # for i in np.linspace(0,93,32):
# #     i = int(i)
# #     params.append(non_trainable_params[i+2])
# #     k['new_parameter_range'][non_trainable_params[i+2]] = [0.0, 50.0]
    
# # for i in np.linspace(0,90,31):
# #     i = int(i)
# #     k['set_nontrainable_parameters'][non_trainable_params[i]] = 1.0 # type = bell
# #     k['set_nontrainable_parameters'][non_trainable_params[i+1]] = 2.0 # mode = "BWC (BT)"
# # #     k['set_nontrainable_parameters'][non_trainable_params[i+2]] = 1.0 # Q = 1

# # set nontrainable parameters for highpass, lowshelf, highshelf, lowpass: type filter

# set_nontrainable_parameters[14] = 2 #highpass
# set_nontrainable_parameters[24] = 5 #lowshelf
# set_nontrainable_parameters[314] = 3 #highshelf
# set_nontrainable_parameters[324] = 4 #lowpass

# # Add input_gain and output_gain

# parameters.append(3)
# parameters.append(4)

# new_parameter_range[3] = [0.0, 4.0]
# new_parameter_range[4] = [0.0, 4.0]


# params = len(parameters)
# param_map = {}
# for i, port in enumerate(parameters):
#     param_map[i] = port
# k['params'].append(params)
# k['plugin_uri'].append(plugin_uri)
# k['param_map'].append(param_map)
# k['stereo'].append(stereo)
# k['set_nontrainable_parameters'].append(set_nontrainable_parameters)
# k['new_parameter_range'].append(new_parameter_range)






    

    
    
    
    
    
# k['plugin_uri'] = 'http://lsp-plug.in/plugins/lv2/mb_expander_mono'
# k['param_map'] = {0:13,1:15,2:17,3:4} # Expander
# k['param_map'] = {0:13,1:19,2:12,3:4} # Gate
# k['param_map'] = {0:13,1:12,2:14} # Gate
# k['param_map'] = {0:17} # Expander

# params = [19,20,21,31,32,33,43,44,45,55,56,57] #multiband gate stereo
# params = [19,20,21,31,32,33,43,44,45,55,56,57,22,23,34,35,46,47,58,59] # bands + attack and release
# params = [19,20,21,31,32,33,43,44,45,55,56,57,22,23,34,35,46,47,58,59,15,16] # bands + attack and release +freq split 1/2 and 2/3

# params = [19,20,21,31,32,33,43,44,45,55,56,57,22,23,34,35,46,47,58,59,15,16,17] # bands + attack and release +freq split 1/2 2/3 and 3/4