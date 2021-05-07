#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
from setuptools import setup
from importlib.machinery import SourceFileLoader

with open('README.md') as file:
    long_description = file.read()

version = SourceFileLoader('deepafx.version', 'deepafx/version.py').load_module()

setup(
   name='deepafx',
   version=version.version,
   description='DeepAFX: Use deep learning to control audio effects plugins and perform AI audio/music production.',
   author='Marco A. Martinez Ramirez, Oliver Wang, Paris Smaragdis, Nicholas J. Bryan',
   author_email='no-reply@adobe.com',
   url='http://github.com/adobe-research/deepafx',
   packages=['deepafx'],
   long_description=long_description,
   long_description_content_type='text/markdown',
   keywords='audio music effects fx ai dsp signalprocessing machinelearning deeplearning',
   license='ADOBE RESEARCH LICENSE',
   classifiers=[
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
   install_requires=['numpy>=1.18.4',
                     'librosa==0.7.2',
                     'gpustat>=0.6.0',
                     'numba==0.48',
                     'pyloudnorm>=0.1.0',
                     'matplotlib>=3.2.2',
                     'sox>=1.4.1', 
                     'tensorflow==2.2.0', 
                     'binaryornot'],
)
