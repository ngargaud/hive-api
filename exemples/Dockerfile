FROM ubuntu:22.04

# docker build --rm -t hive_test -f Dockerfile . && docker run --rm -ti --network=host --name=hive_test hive_test

LABEL Version="1.0"
LABEL Maintainer="Nicolas Gargaud <jacen92@gmail.com>"
LABEL Description="test hive API"

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    &&  apt-get install -y apt-utils sudo nano htop tree python3 python3-pip git wget \
    &&  rm -rf /var/lib/apt/lists/*

ARG USER_NAME="dev"
ARG USER_UID="1000"
ENV USER_NAME=${USER_NAME}
ENV USER_UID=${USER_UID}
ENV WORKTREE_DIRECTORY "/home/${USER_NAME}"

RUN adduser --disabled-password --gecos "" ${USER_NAME} \
 && usermod -aG sudo ${USER_NAME} && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
 && echo "${USER_NAME}:${USER_NAME}" | chpasswd \
 && usermod -u ${USER_UID} ${USER_NAME}

RUN su ${USER_NAME} -c "pip install --upgrade --user pip"
RUN su ${USER_NAME} -c "pip install --upgrade --user setuptools"
RUN su ${USER_NAME} -c "pip install --upgrade --user pillow"
RUN su ${USER_NAME} -c "pip install --upgrade --user git+https://github.com/ngargaud/hive-api.git"
COPY --chown=${USER_NAME}:${USER_NAME} use_hive.py ${WORKTREE_DIRECTORY}/use_hive.py

# patch httpx to disable ssl verification
# ENV HTTPX_NO_SSL=True
RUN sed -i 's/if self.verify:/if self.verify and not os.environ.get("HTTPX_NO_SSL", "").lower() in ["true", "on", "1"]:/' /home/${USER_NAME}/.local/lib/python3.10/site-packages/httpx/_config.py

USER ${USER_NAME}
WORKDIR ${WORKTREE_DIRECTORY}
CMD ["python3", "/home/dev/use_hive.py", "--hive=https://192.168.0.21"]
