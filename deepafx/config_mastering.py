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
k['encoder'] = 'inception' # inception or mobilenet

# Define paths
k['path_audio'] = '/home/code-base/scratch_space/mastering/mixing_secrets_mastering_22050/'
k['path_models'] = '/home/code-base/scratch_space/models/mastering/'


# DAFX constants
k['output_length'] = 1024 # Length of output frame
k['hop_samples'] = 1024 # Length of hop size, for non-overlapping frames it should be equal to output_length
k['gradient_method'] = 'spsa' # or 'spsa'
k['compute_signal_gradient'] = False
k['multiprocess'] = True
k['greedy_dafx_pretraining'] = True # Enables progressive training
k['default_pretraining'] = True # Enables default initialization of parameter values for training


# Single DAFx:

# # Multiband compressor

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

# # # Multiband Compressor

plugin_uri = 'http://calf.sourceforge.net/plugins/MultibandCompressor'
stereo = True
parameters = [19,20,23,30,31,34,41,42,45,52,53,56,15,16,17,5] # threshold, ratio, makeupgain, freqs, input gain, 
# parameters = [19,20,23,24,30,31,34,35,41,42,45,46,52,53,56,57,15,16,17,5,6] # threshold, ratio, makeupgain, knee, freqs, input output  gain
# parameters = [19,20,24,30,31,35,41,42,46,52,53,57,15,16,17] # threshold, ratio, makeupgain, knee, freqs, input output  gain
# parameters = [19,20,21,22,23,24,30,31,32,33,34,35,41,43,44,42,45,46,52,53,54,55,56,57,15,16,17] # threshold, ratio, makeupgain, knee, attack release

# parameters = [19,20,21,22,23,30,31,32,33,34,41,43,44,42,45,52,53,54,55,56,15,16,17] # threshold, ratio, makeupgain, attack release

# parameters = [19,20,21,22,23,30,31,32,33,34,41,43,44,42,45,52,53,54,55,56,15,16,17,5] # threshold, ratio, makeupgain, attack, release, input gain

set_nontrainable_parameters = {28:0,
                                    39:0,
                                    50:0,
                                    61:0,
                                    21:0.01,
                                    22:0.01,
                                    32:0.01,
                                    33:0.01,
                                    43:0.01,
                                    44:0.01,
                                    54:0.01,
                                    55:0.01,
                                   }

new_parameter_range = {15:[10.0, 400.0],
                       16:[400.0, 3000.0],
                       17:[3000.0,10000.0],
                       23:[1.0, 5.0],
                       34:[1.0, 5.0],
                       45:[1.0, 5.0],
                       56:[1.0, 5.0],
#                        21:[0.010000, 1000.0],
#                        22:[0.010000, 1000.0],
#                        32:[0.010000, 1000.0],
#                        33:[0.010000, 1000.0],
#                        43:[0.010000, 1000.0],
#                        44:[0.010000, 1000.0],
#                        54:[0.010000, 1000.0],
#                        55:[0.010000, 1000.0],
                           5:[0.015625,4],
#                            6:[0.015625,5]
                           }

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

# # Parametric EQ


# plugin_uri = 'http://calf.sourceforge.net/plugins/Equalizer8Band'
# stereo = True
# parameters = [16,22,23,25,26,28,29,30,32,33,34,36,37,38,40,41,42,5,6] # freq, gain, Q * 4 bands, freq, gain * 4 bands, input gain, output gain


# set_nontrainable_parameters = {15:1, # HP
#                                18:0, # LP
#                                21:1, # LS
#                                24:1, # HS
#                                27:1, # F1
#                                31:1, # F2
#                                35:1, # F3
#                                39:1, # F4
#                                16:30,
#                        19:11000,
#                        22:1,
#                        23:100,
#                        25:1,
#                        26:5000,
#                        28:1,
#                        29:200,
#                        30:1,
#                        32:1,
#                        33:500,
#                        34:1,
#                        36:1,
#                        37:2000,
#                        38:1,
#                        40:1,
#                        41:4000,
#                        42:1,
#                          }

# new_parameter_range = {16:[10.0, 100.0],
#                        22:[0.015625, 5],
#                        23:[10.0, 1000.0],
#                        25:[0.015625, 5],
#                        26:[5000.0, 11000.0],
#                        28:[0.015625, 5],
#                        29:[10.0, 11000.0],
#                        30:[0.1, 20.0],
#                        32:[0.015625, 5],
#                        33:[10.0, 11000.0],
#                        34:[0.1, 20.0],
#                        36:[0.015625, 5],
#                        37:[10.0, 11000.0],
#                        38:[0.1, 20.0],
#                        40:[0.015625, 5],
#                        41:[10.0, 11000.0],
#                        42:[0.1, 20.0],
#                        5:[0.015625, 2.0],
#                        6:[0.015625, 2.0]
#                            }

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

# # Limiter-CALF



# plugin_uri = 'http://calf.sourceforge.net/plugins/Limiter'
# stereo = True

# parameters = [15, 17, 21, 5, 6] # limit, release, ASC coeff, input gain, output gain,

# # parameters = [15]
# set_nontrainable_parameters = {16:5.0}
                                  
# new_parameter_range = {
#                         17:[1.,50.0],
#                         5:[0.015625, 2.0],
#                         6:[0.015625, 2.0]
#                         }

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

# # 32 Band Graph EQ

plugin_uri = 'http://lsp-plug.in/plugins/lv2/graph_equalizer_x32_mono'
stereo = False

parameters = [18]
for i in range(31):
    parameters.append(parameters[i]+5)
    
new_parameter_range = {}
for i in parameters:
    new_parameter_range[i] = [0.015850, 5.0]
    
# parameters.append(3)
parameters.append(4)

# k['new_parameter_range'][3] = [0.0, 4.0]
new_parameter_range[4] = [0.0, 4.0]


set_nontrainable_parameters = {6:4}

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





# Limiter -lsp

plugin_uri = 'http://lsp-plug.in/plugins/lv2/limiter_mono'
stereo = False

# parameters = [7,8,10,11,12] # th, knee, lookahead, attack, release
# parameters = [7,10] # th, lookahead,
parameters = [7] # th,
# 
# parameters = [15]
set_nontrainable_parameters = {}
                                  
new_parameter_range = {}

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



