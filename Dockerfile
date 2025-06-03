###########################################
# Official Image Ubuntu with OpenSSH server
# Allow SSH connection to the container
# Installed: openssh-server, mc, htop, zip,
# tar, iotop, ncdu, nano, vim, bash, sudo
# for net: ping, traceroute, telnet, host,
# nslookup, iperf, nmap
###########################################


FROM ubuntu:20.04

WORKDIR /root/
USER root
RUN mkdir -p /tutorial/

RUN apt-get update 
RUN apt-get install -y software-properties-common curl
RUN apt-get install -y git

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

RUN apt-get install -y nano pciutils lshw

RUN apt-get install -y build-essential libsqlite3-dev

RUN apt-get install -y python3.13-full python3.13-dev python3.13-venv \
    && python3.13 -m ensurepip --upgrade \
    && python3.13 -m pip install --upgrade pip setuptools wheel build

RUN rm -f /usr/bin/python \
    && rm -f /usr/bin/pip \
    && rm -f /usr/bin/python3 \
    && rm -f /usr/bin/pip3

RUN ln -s /usr/bin/python3.13 /usr/bin/python \
     && ln -s /usr/bin/python3.13 /usr/bin/python3 \
     && ln -s /usr/local/bin/pip3.13 /usr/bin/pip \
     && ln -s /usr/local/bin/pip3.13 /usr/bin/pip3

RUN pip install jupyter \
    && pip install ipykernel -U --user --force-reinstall \
    && pip install ipywidgets \
    && pip install matplotlib \
    && pip install numpy pandas \
    && pip install pyzotero \
    && pip install langchain \
    && pip install langchain-community \
    && pip install langchain-chroma \
    && pip install langgraph \
    && pip install pypdf \
    && pip install langfuse


WORKDIR /usr/local/src/

RUN git clone https://github.com/coleifer/pysqlite3.git 

WORKDIR /usr/local/src/pysqlite3/

RUN python setup.py build_full
RUN python setup.py install



RUN cp -r /usr/local/src/pysqlite3/build/lib.linux-x86_64-cpython-313/pysqlite3/* /usr/lib/python3.13/sqlite3/

# Install VSCODE server
RUN curl -fsSL https://code-server.dev/install.sh | sh
COPY vscode-server-config.yaml /root/.config/code-server/config.yaml

# install VSCODE extensions
RUN code-server --install-extension ms-python.python
RUN code-server --install-extension ms-python.debugpy
RUN code-server --install-extension ms-toolsai.jupyter
RUN code-server --install-extension ms-toolsai.jupyter-keymap	
RUN code-server --install-extension ms-toolsai.jupyter-renderers
RUN code-server --install-extension ms-toolsai.vscode-jupyter-cell-tags
RUN code-server --install-extension ms-toolsai.vscode-jupyter-slideshow
    
# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# EXPOSE 22
# EXPOSE 4000
EXPOSE 5000

# # enable bash as default shell
SHELL [ "/bin/bash", "-c" ]
ENTRYPOINT [ "/bin/bash" ]
CMD [ "code-server" ]

