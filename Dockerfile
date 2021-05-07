#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
FROM tensorflow/tensorflow:2.2.0-jupyter
# FROM tensorflow/tensorflow:2.2.0-gpu-jupyter

LABEL maintainer "Nick Bryan <nibryan@adobe.com>"

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y wget && \
    apt-get install unzip && \
    apt-get install -y libsm6 libxext6 libfontconfig1 libxrender1 libglib2.0-0 supervisor python-dev swig ssh libcap2-bin nano && \
    mkdir -p /opt/aws && \
    cd /opt/aws && \
    wget https://s3.amazonaws.com/aws-cli/awscli-bundle.zip && \
    unzip awscli-bundle.zip && \
    python3 awscli-bundle/install -i /usr/local/aws -b /usr/local/bin/aws && \
    rm -rf awscli-bundle.zip awscli-bundle && \
    pip install -U flake8 && \
    pip install -U jsonschema && \
    pip install opencv-python && \
    pip install -U flask && \
    pip install -U gunicorn && \
    pip install -U gevent && \
    pip install -U protobuf && \
    pip install -U newrelic && \
    pip install -U boto3 scipy && \
    apt-get autoclean -y && \
    apt-get autoremove -y && \
    rm -rf /tmp/* /var/tmp/* && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /

# Upgrade pip
RUN /bin/bash -c "python -m pip install --upgrade pip"

# Install the JupyterLab IDE
RUN /bin/bash -c "pip install jupyterlab"


# INSTALL CODE SERVER via https://github.com/cdr/code-server/issues/2341#issuecomment-740892890
RUN /bin/bash -c "curl -fL https://github.com/cdr/code-server/releases/download/v3.8.0/code-server-3.8.0-linux-amd64.tar.gz | tar -C /usr/local/bin -xz"
RUN /bin/bash -c "mv /usr/local/bin/code-server-3.8.0-linux-amd64 /usr/local/bin/code-server-3.8.0"
RUN /bin/bash -c "ln -s /usr/local/bin/code-server-3.8.0/bin/code-server /usr/local/bin/code-server"

# Install Python extension 
RUN /bin/bash -c "wget https://github.com/microsoft/vscode-python/releases/download/2020.10.332292344/ms-python-release.vsix \
 		&& code-server --install-extension ./ms-python-release.vsix || true"

# Install C++ extension
RUN /bin/bash -c "wget https://github.com/microsoft/vscode-cpptools/releases/download/1.1.3/cpptools-linux.vsix  \
		&& code-server --install-extension ./cpptools-linux.vsix || true"

# Set VS Code password to None
#RUN /bin/bash -c "sed -i.bak 's/auth: password/auth: none/' ~/.config/code-server/config.yaml"
COPY docker_scripts/code-server-config.yaml /root/.config/code-server/config.yaml

# Fix broken python plugin # https://github.com/cdr/code-server/issues/2341
RUN /bin/bash -c "mkdir -p ~/.local/share/code-server/ && mkdir -p ~/.local/share/code-server/User"
COPY docker_scripts/settings.json /root/.local/share/code-server/User/settings.json 

# Supervisor setup
RUN apt-get update && apt-get install -y supervisor openssh-client
RUN mkdir -p /var/log/supervisord
COPY docker_scripts/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY docker_scripts/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

 
########## START RESEARCH INSTALLS ##########

# Uncomment to install conda for doing dev
# RUN /bin/bash -c "wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda.sh && \
#                   chmod +x ./miniconda.sh && \
#                   ./miniconda.sh -b -p /opt/conda && \
#                   rm ./miniconda.sh && \
#                   ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh"
#ENV PATH /opt/conda/bin:$PATH

# Update
RUN /bin/bash -c "apt-get update --fix-missing"
RUN /bin/bash -c "pip uninstall matplotlib -y; pip uninstall kiwisolver -y; pip install -U protobuf==3.8.0"

# LV2 plugin installs
RUN /bin/bash -c "apt-get install apt-utils -y && apt-get install pkg-config -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install ubuntustudio-audio-plugins -y && apt-get install libsndfile-dev -y \
    && wget https://sourceforge.net/projects/lsp-plugins/files/lsp-plugins/1.1.19/Linux-x86_64/lsp-plugins-lv2-1.1.19-Linux-x86_64.tar.gz -P /home/code-base/ \
    && tar -C /home/code-base/ -xvf /home/code-base/lsp-plugins-lv2-1.1.19-Linux-x86_64.tar.gz \
    && cp -rf /home/code-base/lsp-plugins-lv2-1.1.19-Linux-x86_64/usr/local/lib/lv2/lsp-plugins.lv2 /usr/lib/lv2/ \
    && rm -rf /home/code-base/lsp-plugins-lv2-1.1.19-Linux-x86_64.tar.gz \
    && rm -rf /home/code-base/lsp-plugins-lv2-1.1.19-Linux-x86_64 \
    && apt-get install dh-autoreconf -y \
    && apt-get install meson -y \
    && apt-get install psmisc -y \
    && apt-get install sox -y \
    && apt-get install libsox-fmt-mp3 -y"

# Install LV2 tools
RUN /bin/bash -c "apt-get install lv2proc -y \
    && apt-get install lilv-utils -y \
    && apt-get install lv2-dev -y \
    && apt-get install liblilv-dev -y"

ENV RESEARCH_PACKAGE_NAME="deepafx"
ENV CODE_BASE="/home/code-base/runtime"

# Install research code package
COPY . $CODE_BASE
RUN /bin/bash -c "pip install -e $CODE_BASE --upgrade"
RUN /bin/bash -c "mkdir -p /home/code-base/scratch_space"

########## END RESEARCH INSTALLS ##########


EXPOSE 8080 8888 8887 443

ENTRYPOINT ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]