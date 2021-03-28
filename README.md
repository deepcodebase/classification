# Classification Codebase

This repo implements a simple PyTorch codebase for training classification models with powerful tools including Docker, PyTorchLightning, and Hydra.

## Requirements

- nvidia-docker
- docker-compose


## Setup

### Build the environment

We use docker to run all experiemnts. Before running any codes, you should check `docker-compose.yml` first. The defualt setting is shown as below:

``` yaml
version: "3.9"
services:
    playground:
        container_name: playground
        build:
            context: docker/
            dockerfile: Dockerfile.local
            args:
                - USER_ID=${UID}
                - GROUP_ID=${GID}
                - USER_NAME=${USER_NAME}
        image: pytorch_local
        environment:
            - TZ=Asia/Shanghai
            - TORCH_HOME=/data/torch_model
        ipc: host
        hostname: docker
        working_dir: /code
        command: ['sleep', 'infinity']
        volumes:
            - .:/code
            - /data1/data:/data
            - /data2/data/train_log/outputs:/outputs
```

You should change the `volumes` to:
- mount your dataset folders to `/data`,
- and mount a folder for `/outputs` (training logs will be written to this folder)

Next, simply run:

``` sh
python core.py env prepare
```

This command will first build an image based on `/docker/Dockerfile.local` and then luanch a container based on this image.


### Enter the environment

Simply run:

``` sh
python core.py env
```

The default user is the same as the host to avoid permission issues. And of course you can enter the container with root:

``` sh
python core.py env --root
```

### Change the environment

Basiclly, there are four config files:

- `/docker/Dockerfile.pytorch` defines basic environments including cuda, cudnn, nccl, conda, torch, etc. This image has been build at [`deepbase/pytorch`](https://hub.docker.com/r/deepbase/pytorch). By default, you don't need to change this.
- `/docker/Dockerfile.local` defines the logic of building the local image. For example, install packages defined in `requirements.txt`.
- `/docker/requirements.txt` defines the python packages you want to install.
- `/docker-compose.yml` defines the setting of running the container. For example, the volumes, timezone, etc.


After changing the settings as you want, you can rebuild the local image by running:


``` sh
python core.py env prepare --build
```


## Training

Enter the environment and run:

``` sh
python train.py
```

### Suggestions

Reading the offical documents of Hydra and PyTorchLightning to know more:
- [Hydra](https://hydra.cc/docs/intro): Very powerful and convenient configuration system and more.
- [PyTorchLightning](https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html): You almost only need to write codes for models and data. Say goodbye to codes for pipelines, mixed precision, logging, etc.
