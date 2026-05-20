FROM gitlab-master.nvidia.com:5005/minyu/images:hstu-sm120-debug

RUN export DEBIAN_FRONTEND=noninteractive \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
        sudo \
        openssh-server \
        socat

# Install sandbox runtime so seccomp filter artifacts are available.
# If npm is missing in the base image, install nodejs/npm first.
RUN export DEBIAN_FRONTEND=noninteractive \
  && if ! command -v npm >/dev/null 2>&1; then \
       apt-get update && apt-get install -y --no-install-recommends nodejs npm; \
     fi \
  && npm install -g @anthropic-ai/sandbox-runtime

RUN addgroup --gid 103932 minyu
RUN groupmod --gid 3000 dip
RUN addgroup --gid 30 hardware
RUN adduser --disabled-password --gecos GECOS -u 103932 -gid 30 minyu
RUN adduser minyu sudo
RUN usermod -a -G dip minyu
RUN usermod -a -G dip root
RUN usermod -a -G root minyu
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Configure SSH
RUN sed -i 's/^#\?\s*PasswordAuthentication.*$/PasswordAuthentication yes/' /etc/ssh/sshd_config
RUN sed -i 's/^#\?\s*AllowAgentForwarding.*$/AllowAgentForwarding yes/' /etc/ssh/sshd_config
RUN sed -i 's/^#\?\s*AllowTcpForwarding.*$/AllowTcpForwarding yes/' /etc/ssh/sshd_config

USER minyu

RUN sudo apt update
RUN sudo apt install -y --no-install-recommends gnupg
#RUN sudo echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
#RUN sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN sudo apt update
RUN sudo apt-get install -y --no-install-recommends bubblewrap

RUN git config --global user.name "minyu"
RUN git config --global user.email "minyu@nvidia.com"
ENV CCACHE_DIR=/workspace/.ccache
RUN export PATH=$PATH:/home/minyu/.local/bin

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sudo chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
