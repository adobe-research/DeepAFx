#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
import numpy as np

k = {}

k['sr'] = 22050 # Sampling rate
k['num_samples'] = 40960 # Length of total input frame in samples
k['batch_size'] = 100 
k['steps_per_epoch'] = 1000
k['epochs'] = 1000
k['patience'] = 25
k['encoder'] = 'inception' # inception or mobilenet

k['path_audio'] = '/home/code-base/scratch_space/distortion/6176ChannelStrip_22050/'
k['x_recording'] = 'dry/'
k['y_recording'] = 'preamp/'
k['path_models'] = '/home/code-base/scratch_space/models/distortion/' # Path to save models

# DAFX constants
k['output_length'] = 1024 # Length of output frame
k['hop_samples'] = 1024 # Length of hop size, for non-overlapping frames it should be equal to output_length
k['gradient_method'] = 'spsa' # or 'spsa'
k['compute_signal_gradient'] = False
k['multiprocess'] = True
k['greedy_dafx_pretraining'] = True # Enables progressive training
k['default_pretraining'] = True # Enables default initialization of parameter values for training

# MULTIPLE DAFx:


k['params'] = []
k['plugin_uri'] = []
k['param_map'] = []
k['stereo'] = []
k['set_nontrainable_parameters'] = []
k['new_parameter_range'] = []

# Tube Emulator

# plugin_uri = 'http://invadarecords.com/plugins/lv2/tube/mono'
# stereo = False
# parameters = [1,2,4] # bands +freq split 1/2 2/3 

# set_nontrainable_parameters = {}
# new_parameter_range = {}

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

# # Multiband compressor

plugin_uri = 'http://calf.sourceforge.net/plugins/MultibandCompressor'
stereo = True
# params = [19,20,30,31,41,42,52,53,15,16,17] # threshold, ratio, freqs.
# params = [19,20,23,30,31,34,41,42,45,52,53,56,15,16,17] # threshold, ratio, makeupgain, freqs.
# params = [19,20,23,30,31,34,41,42,45,52,53,56,15,16,17,5] # threshold, ratio, makeupgain, freqs, out gain
# params = [19,20,23,30,31,34,41,42,45,52,53,56,15,16,17,5,6] # threshold, ratio, makeupgain, freqs, input output gain gain

parameters = [19,20,23,24,30,31,34,35,41,42,45,46,52,53,56,57,15,16,17,5,6] # threshold, ratio, makeupgain, knee, freqs, input output  gain

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
                            16:[300.0, 3000.0],
                           17:[3000.0,8000.0],
                           5:[0.015625,2],
                           6:[0.015625,2]
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


# Limiter -lsp

# plugin_uri = 'http://lsp-plug.in/plugins/lv2/limiter_mono'
# stereo = False

# parameters = [7] # 

# # parameters = [15]
# set_nontrainable_parameters = {}
                                  
# new_parameter_range = {}

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