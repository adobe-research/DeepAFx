#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
import math
import sys
import wave
import numpy
import soundfile
import os
import librosa
import time
import scipy
import multiprocessing as mp

from deepafx.lilv import lilv


class LV2_Plugin:
    """A Linux Audio Plugin Version 2 (LV2) python object.
    
    This class allows you to create an LV2 plugin in Python, set parameters 
    on the effect, render audio through the processor, and get the output.
    

    """
    
    def __init__(self, plugin_uri, sr, hop_samples=64, verbose=False, mono=False):
        """Constructor that creates the internal LV2 C++ object.
        
        Parameters
        ----------
        plugin_uri : str
            The LV2 plugin uri
        sr : int
            Sampling rate of the plugin
        hop_samples : int
            Plugin host render hop size in samples.
        verbose : bool
            Flag to enable verbose log printing
        mono: bool
            Falt to run stereo plugins as mono
        
        """
        
        # Create the lilv plugin finder
        world = lilv.World()
        ns    = world.ns
        world.load_all()
        
        # Find plugin
        plugin_uri_node = world.new_uri(plugin_uri)
        plugins         = world.get_all_plugins()
        if plugin_uri_node not in plugins:
            print("Unknown plugin `%s'" % plugin_uri)
            sys.exit(1)
        
        plugin      = plugins[plugin_uri_node]
        n_audio_in  = plugin.get_num_ports_of_class(ns.lv2.InputPort,  ns.lv2.AudioPort)
        n_audio_out = plugin.get_num_ports_of_class(ns.lv2.OutputPort, ns.lv2.AudioPort)
        if n_audio_out == 0:
            print("Plugin has no audio outputs\n")
            sys.exit(1)
            
        if verbose: print('plugin name', plugin.get_name())
        
        instance = lilv.Instance(plugin, sr)
        
        # Setup preset map
        control_port_to_value_type = {}
        for index in range(plugin.get_num_ports()):
            port = plugin.get_port_by_index(index)
#             if port.is_a(ns.lv2.InputPort):
            if port.is_a(ns.lv2.AudioPort):
                pass
            elif port.is_a(ns.lv2.ControlPort) and port.is_a(ns.lv2.InputPort):  
                t = numpy.float32
                if port.get(ns.lv2.default).is_float():
                    default = float(port.get(ns.lv2.default))
                    t = numpy.float32
                elif port.get(ns.lv2.default).is_bool():
                    default = bool(port.get(ns.lv2.default))
                    t = numpy.bool
                elif port.get(ns.lv2.default).is_int():
                    default = int(port.get(ns.lv2.default))
                    t = numpy.int32
                elif port.get(ns.lv2.default).is_str():
                    default = port.get(ns.lv2.default)
                    raise('String parameter is not going to work')
                else:
                    default = 0
                    t = numpy.float32
                    raise('Unknown port type')
                control_port_to_value_type[index] = (default, t)
            
            elif port.is_a(ns.lv2.ControlPort) and port.is_a(ns.lv2.OutputPort):  
                
                default = 0
                t = numpy.float32
                if port.get_node().is_float():
                    default = port.get(ns.lv2.default)
                    t = numpy.float32
                elif port.get_node().is_bool():
                    default = port.get(ns.lv2.default)
                    t = numpy.bool
                elif port.get_node().is_int():
                    default = port.get(ns.lv2.default)
                    t = numpy.int32
                control_port_to_value_type[index] = (default, t)             
                
                
            else:
                # Custom port type
                pass
                #print("Unhandled port type")
                #raise ValueError("Unhandled port type")

        # Verify the plugin
        try:
            plugin.verify()
        except:
            raise('ERROR: Could not verify the plugin')
 
            
        # Make port I/O buffers (audio & control)
        buffer_size = hop_samples #1024
        audio_input_buffers    = {}
        audio_output_buffers   = {}
        control_input_buffers  = {}
        control_output_buffers = {}
        for index in range(plugin.get_num_ports()):
            port = plugin.get_port_by_index(index)

            # Connect input ports
            if port.is_a(ns.lv2.InputPort):
                if port.is_a(ns.lv2.AudioPort):
                    audio_input_buffers[index] = numpy.array(numpy.zeros((buffer_size,)), numpy.float32)
                    instance.connect_port(index, audio_input_buffers[index])
                elif port.is_a(ns.lv2.ControlPort):  
                    default, t = control_port_to_value_type[index]
                    control_input_buffers[index] = numpy.array([default], numpy.float32)
                    instance.connect_port(index, control_input_buffers[index])
                else:
                    # Custom port
                    pass
                    #print("Unhandled port type", index, port)
                    #raise ValueError("Unhandled port type")

            # Connect output ports
            elif port.is_a(ns.lv2.OutputPort):
                if port.is_a(ns.lv2.AudioPort):
                    audio_output_buffers[index] = numpy.array([0] * buffer_size, numpy.float32)
                    instance.connect_port(index, audio_output_buffers[index])
                elif port.is_a(ns.lv2.ControlPort):
                    default, t = control_port_to_value_type[index]
                    control_output_buffers[index] = numpy.array([0], numpy.float32)
                    instance.connect_port(index, control_output_buffers[index])
                else:
                    # Custom port
                    pass
                    #print("Unhandled port type", index, port)
                    #raise ValueError("Unhandled port type")
                    
    
        # MUST be called before run
        instance.activate()
        
        # Assign to class
        self.verbose = verbose
        self.plugin = plugin
        self.instance = instance
        self.sr = sr
        self.ns = ns
        self.buffer_size = buffer_size
        self.audio_input_buffers = audio_input_buffers    
        self.audio_output_buffers = audio_output_buffers
        self.control_input_buffers = control_input_buffers
        self.control_output_buffers = control_output_buffers
        self.mono = mono
        

        
    def __del__(self):
        """Destructor needed to deallocate the internal LV2 object."""
        
        # Deactivate (matches activate)
        self.instance.deactivate()
        
    def reset(self):
        self.instance.activate()
        
        
    def print_plugin_info(self):
        """Print information about the specific LV2 plugin."""
        
    
        print('Plugin Name:', self.plugin.get_name())
        print('Plugin Project:', self.plugin.get_project())
        print('Plugin URI:', self.plugin.get_uri())
        print('Library URI:', self.plugin.get_library_uri())
        print('Number of ports:', self.plugin.get_num_ports())
        print('Has Latency:', self.plugin.has_latency())
        print('Class:', self.plugin.get_class())
        print('Author Name:', self.plugin.get_author_name())
        print('Author Email:', self.plugin.get_author_email())
        print('Author Homepage:', self.plugin.get_author_homepage())
        
        # Print input audio ports
        if self.verbose:
            for index in range(self.plugin.get_num_ports()):
                port = self.plugin.get_port_by_index(index)
                print(port.get_name())
                print('\tindex:', index)
                print('\tsymbol:', port.get_symbol())
                print('\tDefault:', port.get_range()[0])
                print('\tMinimum:', port.get_range()[1])
                print('\tMaximum:', port.get_range()[2])
                print('\tAudio Port:', port.is_a(self.ns.lv2.AudioPort))
                print('\tControl Port:', port.is_a(self.ns.lv2.ControlPort))
                print('\tInput Port:', port.is_a(self.ns.lv2.InputPort))
                print('\tOutput Port:', port.is_a(self.ns.lv2.OutputPort))
                if port.is_a(self.ns.lv2.ControlPort):
                    n = port.get(self.ns.lv2.default)
                    try:
                        print('\tNode literal:', n.is_literal())
                        print('\tNode URI:', n.is_uri())
                        print('\tNode blank:', n.is_blank())
                        print('\tNode bool:', n.is_bool())
                        print('\tNode float:', n.is_float())
                        print('\tNode int:', n.is_int())
                        print('\tNode string:', n.is_string())

                    except Exception as e:
                        print('\t',e)
                # Audio ports are always floats

    def set_param(self, port_id, value):
        """Set a parameter given the port index and value.
        
        Parameters
        ----------
        port_id : int
            The LV2 parameter port id 
        value : float
            The LV2 parameter port value
        """    
        
        before = self.control_input_buffers[port_id][0] 
        self.control_input_buffers[port_id][0] = value
        
    def get_param_range(self, port_id):
        """Return the port rangge tuble values (default, min, max).
        
        Parameters
        ----------
        port_id : int
            The LV2 parameter port id 
        """
                 
        port = self.plugin.get_port_by_index(port_id)
        return port.get_range()

    
        
    def runs(self, input_time):
        """Process a signal through the LV2 plugin buffer by buffer.
    
        Parameters
        ----------
        input_time : np.array
            Input audio signal. 
            Shape is num_channels x num_frames (e.g. 1 x 100000 for mono)
            
        Return Values
        ----------
        output_time : np.array
            Output processed audio signal. Shape is num_channels x num_frames
        
        """
        noise = 0.00001*numpy.random.normal(0,
                                            1,
                                            size=(input_time.shape[0],input_time.shape[1]))
        channels = input_time + noise
        
        # Get total number of samples (per channel) to process
        nframes = channels.shape[1]
   
        ## Process the audio buffer by buffer     
        left = 0
        right = self.buffer_size

        # Define output buffer to save to disk
        output = numpy.empty((len(self.audio_output_buffers), len(channels[0])))
        while True:
            
            if right > nframes:
                break
                
            # Copy in the input. 
            for i, c in enumerate(self.audio_input_buffers):
                self.audio_input_buffers[c][0:self.buffer_size] = channels[i][left:right]

            # Process C++ LV2
            self.instance.run(self.buffer_size)
            
            # Copy out the output
            for i, c in enumerate(self.audio_output_buffers):
                output[i][left:right] = self.audio_output_buffers[c][0:self.buffer_size]
    
            # Update indices
            left = left + self.buffer_size
            right = right + self.buffer_size
            
        output = numpy.clip(output, -100.0, 100.0)
    
        return output + noise
    
    def runs_stereo(self, input_time):
        """Process a signal through the LV2 plugin buffer by buffer. 
        Use when input and output are is mono but plugin works on stereo
    
        Parameters
        ----------
        input_time : np.array
            Input mono audio signal. 
            Shape is num_channels x num_frames (e.g. 1 x 100000 for mono)
            
        Return Values
        ----------
        output_time : np.array
            Output mono processed audio signal. Shape is num_channels x num_frames
        
        """
        
        
        noise = 0.00001*numpy.random.normal(0,
                                            1,
                                            size=(input_time.shape[0],input_time.shape[1]))
        channels = input_time + noise
        
        # Get total number of samples (per channel) to process
        nframes = channels.shape[1]
   
        ## Process the audio buffer by buffer     
        left = 0
        right = self.buffer_size

        # Define output buffer to save to disk
        output = numpy.empty((1, len(channels[0])))
        
        while True:
            
            if right > nframes:
                break
#           TODO: This assumes that left input and output channels are ports 0 and 2. 
            # Check if not sending data via right channel brings problems.
            # Copy in the input. 
#             for i, c in enumerate(self.audio_input_buffers):
            self.audio_input_buffers[0][0:self.buffer_size] = channels[0][left:right]

            # Process C++ LV2
            self.instance.run(self.buffer_size)
            
            # Copy out the output
#             for i, c in enumerate(self.audio_output_buffers):
            output[0][left:right] = self.audio_output_buffers[2][0:self.buffer_size]
    
            # Update indices
            left = left + self.buffer_size
            right = right + self.buffer_size
            
        output = numpy.clip(output, -100.0, 100.0)
            
        return output + noise

 
    def run_from_file(self, wav_in_path, wav_out_path):
        """Process an audio wavefile through the LV2 plugin buffer by buffer.
        
        Parameters
        ----------
        wav_in_path : str
            filepath of an input wavefile
        wav_out_path: str
            filepath of an output wavefile to save to
        
        """
        
        t = time.time()
        
        if self.verbose: print('%s => %s @ %d Hz' % (wav_in_path, wav_out_path, self.sr))
        
        # Read in the audio file
        channels, sr = librosa.core.load(wav_in_path, sr=self.sr, mono=False)
        if len(channels.shape) == 1:
            channels = numpy.expand_dims(channels, axis=0)
            
        # Run samples 
        output = self.runs(channels)
            
        # save the output
        soundfile.write(wav_out_path, output.T, sr, subtype='FLOAT')
        
        if self.verbose: print("Time elapsed:", time.time() - t)
            
    def get_latency_plugin(self, stereo=False):
        
        if not stereo:
            x = numpy.zeros(4*self.sr)
            x[int(self.sr)] = 1
            y = self.runs(numpy.expand_dims(x,0))
            y = numpy.squeeze(y,axis=0)
        else:
            x = numpy.zeros((2,4*self.sr))
            x[0][int(self.sr)] = 1
            y = self.runs(x)
            y = y[0]
            x = x[0]
            
        A = scipy.fftpack.fft(x)
        B = scipy.fftpack.fft(y)
        Ar = -A.conjugate()
        time_shift = numpy.argmax(numpy.abs(scipy.fftpack.ifft(Ar*B)))
        

        return time_shift
    
    def reset_plugin_state(self, samples, stereo=False):

        noise = 0.01*numpy.random.normal(0, 1, size=samples)
        noise = numpy.expand_dims(noise,0)
        
        if stereo:
            out = self.runs_stereo(noise)
        else:
            out = self.runs(noise)
            
            
            
class LV2_Plugin_Chain:
    """A Linux Audio Plugin Version 2 (LV2) python object.
    
    This class allows you to create a chain of LV2 plugins in Python, set parameters 
    on the effects, render audio through the processor, and get the output.
    

    """
    
    def __init__(self, plugin_uri_list, stereo_list, sr, hop_samples=64, verbose=False):
        """Constructor that creates the internal LV2 C++ object.
        
        Parameters
        ----------
        plugin_uri : list(str)
            The LV2 plugin uri
        sr : int
            Sampling rate of the plugin
        hop_samples : int
            Plugin host render hop size in samples.
        verbose : bool
            Flag to enable verbose log printing
        mono: bool
            Falt to run stereo plugins as mono
        
        """
        self.plugins = []
        for plugin_uri in plugin_uri_list:
             self.plugins.append(LV2_Plugin(plugin_uri, sr, hop_samples=hop_samples, verbose=verbose))  

        # Assign to class
        self.verbose = verbose
        self.sr = sr
        self.stereo = stereo_list
        

        
    def __del__(self):
        """Destructor needed to deallocate the internal LV2 object."""
        
        # Deactivate (matches activate)
        for plugin in self.plugins:
            plugin.instance.deactivate()
        
    def reset(self):
        for plugin in self.plugins:
            plugin.instance.activate()
        
        
    def print_plugin_info(self):
        """Print information about the specific LV2 plugin."""
        for plugin in self.plugins:
            plugin.print_plugin_info()

    def set_param(self, plugin_id, port_id, value):
        """Set a parameter given the port index and value.
        
        Parameters
        ----------
        plugin_id : int
            The plugin ID
        port_id : int
            The LV2 parameter port id 
        value : float
            The LV2 parameter port value
        """    

        self.plugins[plugin_id].set_param(port_id, value)

        
    def get_param_range(self, plugin_id, port_id):
        """Return the port rangge tuble values (default, min, max).
        
        Parameters
        ----------
        plugin_id : int
            The plugin ID
        port_id : int
            The LV2 parameter port id 
        """
        
        return self.plugins[plugin_id].get_param_range(port_id)

    
        
    def runs(self, input_time, greedy_pretraining=0):
        """Process a signal through the LV2 plugin buffer by buffer.
    
        Parameters
        ----------
        input_time : np.array
            Input audio signal. 
            Shape is num_channels x num_frames (e.g. 1 x 100000 for mono)
            
        Return Values
        ----------
        output_time : np.array
            Output processed audio signal. Shape is num_channels x num_frames
        
        """
        noise = 0.00001*numpy.random.normal(0,
                                            1,
                                            size=(input_time.shape[0],input_time.shape[1]))
        
        if greedy_pretraining:
            
            out = input_time.copy() + noise
            for i, plugin in enumerate(self.plugins[:greedy_pretraining]):
                if self.stereo[i]:
                    out_ = plugin.runs_stereo(out)
                else:
                    out_ = plugin.runs(out)
                out = out_.copy() + noise
        
        else:
            out = input_time.copy() + noise
            for i, plugin in enumerate(self.plugins):
                if self.stereo[i]:
                    out_ = plugin.runs_stereo(out)
                else:
                    out_ = plugin.runs(out)
                out = out_.copy() + noise
                
        out = numpy.clip(out, -100.0, 100.0)
    
        return out

            
    def get_latency_plugin(self, stereo=False):
        
        latency = []
        for i, plugin in enumerate(self.plugins):
            latency.append(plugin.latency(stereo=self.stereo[i]))    

        return latency
    
    def reset_plugin_state(self, samples, stereo=False):
        
        noise = 0.01*numpy.random.normal(0, 1, size=samples)
        noise = numpy.expand_dims(noise,0)
        
        self.runs(noise)
        
class Multi_LV2_Plugin:
    
    def __init__(self, num_processes, plugin_uri, sr, 
                 hop_samples=64, 
                 k_pipe=True, 
                 verbose=False):
        
        self.plugin_uri = plugin_uri
        self.sr = sr
        self.hop_samples = hop_samples
        self.verbose = verbose
        self.num_processes = num_processes
        self.k_pipe = k_pipe
        
        # Create a plugin for each process
        self.plugins = {}
        for i in range(num_processes):
            lv2_dafx = LV2_Plugin(plugin_uri, 
                                             sr, 
                                             hop_samples=hop_samples,
                                             verbose=False)
            self.plugins[i] = lv2_dafx
            
            
        procs = {}
        for i in range(self.num_processes):
            
            plugin = self.plugins[i]

            # create a pipe between the worker and main thread 
            if self.k_pipe:
                parent_conn, child_conn = mp.Pipe()
                p = mp.Process(target=Multi_LV2_Plugin.worker_pipe, 
                               args=(plugin, parent_conn, child_conn,)) 
            else:
                parent_conn = mp.Queue()
                child_conn = mp.Queue()
                p = mp.Process(target=Multi_LV2_Plugin.worker_queue, 
                               args=(plugin, parent_conn, child_conn,))             
                
            # running processes 
            p.start() 

            # Store the process, pipes/queues
            procs[i] = [p, parent_conn, child_conn]
            
        self.procs = procs
    
    def __del__(self):
        self.shutdown()
        
    def worker_queue(lv2_dafx, parent_conn, child_conn): 

        while True: 

            # Try to interpret an END kill message
            try:
                if str(msg) == "END": 
                    break
                elif str(msg) == 'signal':
                    output = lv2_dafx.runs(value.transpose())
                    # Return the output signal
                    child_conn.put(output.transpose())
                elif str(msg) == 'set_param':
                    (port_id, v) = value
                    lv2_dafx.set_param(port_id, v)
            except:
                pass



    def worker_pipe(lv2_dafx, parent_conn, child_conn): 
        """ 
        function to print the messages received from other 
        end of pipe 
        """

        while True: 

            # Read messages from the pipe
            msg, value = child_conn.recv() 

            # Try to interpret an END kill message
            try:
                if str(msg) == "END": 
                    break
                elif str(msg) == 'signal':
                    output = lv2_dafx.runs(value.transpose())
                    # Return the output signal
                    child_conn.send(output.transpose())
                elif str(msg) == 'set_param':
                    (port_id, v) = value
                    lv2_dafx.set_param(port_id, v)
            except:
                pass

 


        
    def shutdown(self):
        
        # Send the final kill/END message
        for key in self.procs:
            if self.k_pipe:
                self.procs[key][1].send(("END", 0)) 
            else:
                self.procs[key][1].put(("END", 0)) 
            
        # Join: Wait until processes finish (kills the thread)
        for key in self.procs:
            self.procs[key][0].join() 
            
        # Clear all the plugins
        self.plugins = {}    
        self.procs = {}
        
    
    def print_plugin_info(self, pid=0):
        return self.plugins[pid].print_plugin_info()
    
    def set_param(self, port_id, value, pid=-1):
        if pid < 0:
            for i in self.plugins:
                self.plugins[i].set_param(port_id, value)
        else:       
            self.plugins[pid].set_param(port_id, value)
    
    def get_param_range(self, port_id, pid=0):
        return self.plugins[pid].get_param_range(port_id)
    
    def get_latency_plugin(self, pid=0):
        return self.plugins[pid].get_latency_plugin()
    
    def run_batch(self, input_batch):
        # Loop over the audio batch to launch the processing

        for i in range(input_batch.shape[0]):
            signal = input_batch[i]
            if self.k_pipe:
                self.procs[i][1].send(('signal', signal))
            else:
                self.procs[i][1].put(('signal', signal))

        # Wait for the return message from each process 
        output = numpy.empty_like(input_batch)
        for i in range(input_batch.shape[0]):
            if self.k_pipe:
                output[i] = self.procs[i][1].recv()
            else:
                output[i] = self.procs[i][2].get()
        return output
                
        


TEST_DENOISER = False
TEST_COMPRESSOR = False

if TEST_COMPRESSOR:
    plugin_uri   = 'http://eq10q.sourceforge.net/compressor'
    sr = 44100
    wav_in_path  = '/home/code-base/noisy_speech.wav'
    wav_out_path = '/home/code-base/output-python.wav'
    t = time.time()
    lv2_dafx = LV2_Plugin(plugin_uri, sr, verbose=True)
#     lv2_dafx.print_plugin_info()
    lv2_dafx.set_param(5, 2.0)
    lv2_dafx.run_from_file(wav_in_path, wav_out_path)
    print("lilv_python - Time elapsed:", time.time() - t)  

    ## TEST with lv2proc
    wav_out_path = '/home/code-base/output-lv2proc.wav'
    params = ' -c compmakeup:10 '
    command = 'lv2proc ' + params + ' -i ' + wav_in_path + ' -o ' + wav_out_path + ' ' + plugin_uri
    t = time.time()
    os.system(command)
    print("lv2proc - Time elapsed:", time.time() - t)
    
    
    ## TEST with lv2apply
    wav_out_path = '/home/code-base/output-lv2apply.wav'
    params = ' -c compmakeup 10'
    command = 'lv2apply ' + params + ' -i ' + wav_in_path + ' -o ' + wav_out_path + ' ' + plugin_uri
    t = time.time()
    os.system(command)
    print("lv2apply - Time elapsed:", time.time() - t)
    
    
if TEST_DENOISER:

    ## TEST with lilv_python
    plugin_uri   = 'https://github.com/lucianodato/speech-denoiser' # Doesn't seem to actually denoise
    sr = 44100
    wav_in_path  = '/home/code-base/noisy_speech.wav'
    wav_out_path = '/home/code-base/output-python.wav'
    t = time.time()
    lv2_dafx = LV2_Plugin(plugin_uri, sr, verbose=True)
    #lv2_dafx.print_plugin_info()
    lv2_dafx.set_param(1, -60)
    lv2_dafx.run_from_file(wav_in_path, wav_out_path)
    print("lilv_python - Time elapsed:", time.time() - t)

    ## TEST with lv2proc
    wav_out_path = '/home/code-base/output-lv2proc.wav'
    params = ' -c mix:-60 '
    command = 'lv2proc ' + params + ' -i ' + wav_in_path + ' -o ' + wav_out_path + ' ' + plugin_uri
    t = time.time()
    os.system(command)
    print("lv2proc - Time elapsed:", time.time() - t)

    ## TEST with lv2apply
    wav_out_path = '/home/code-base/output-lv2apply.wav'
    params = ' -c mix -60'
    command = 'lv2apply ' + params + ' -i ' + wav_in_path + ' -o ' + wav_out_path + ' ' + plugin_uri
    t = time.time()
    os.system(command)
    print("lv2apply - Time elapsed:", time.time() - t)