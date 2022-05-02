# Automated Object Detection Training Pipeline
## Prerequisites
docker-compose will need GPU access, see https://docs.docker.com/config/containers/resource_constraints/#gpu

## Installation
Clone this repo in the folder ~/object_detection and then clone [blender-gen](https://github.com/ignc-research/blender-gen) in the folder object_detection and follow the [installation and usage instructions](https://github.com/ignc-research/blender-gen#installation) to create a synthetic dataset.

Build the container with `docker-compose build detection2d`

## Usage
### Train a Base Model Based on Synthetic Images
```
cd ~/object_detection
source detection2d/envs/base.env
docker-compose up
```

### Finetune the Base Model With Real Images
```
cd ~/object_detection
source detection2d/envs/finetune.env
docker-compose up
```

## Acknowledgments
* [mmdetection](https://github.com/open-mmlab/mmdetection)
* [blender-gen](https://github.com/ignc-research/blender-gen)
