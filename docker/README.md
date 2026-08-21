# Docker

## Prerequisites

- Docker with NVIDIA Container Toolkit (`nvidia-docker2`)
- Access to `nvcr.io/nvidia/isaac-lab:3.0.0-beta2` (Isaac Sim 6.0 / Isaac Lab 3.0 beta)
- To push the built image, a container registry of your own (set `ROBOLAB_REGISTRY` to its image path prefix)

## Build

```bash
# Uses git short SHA as tag by default
./docker/build_docker.sh

# Older simulator stacks remain available
./docker/build_docker.sh --isaac50
./docker/build_docker.sh --isaac51

# Custom tag
./docker/build_docker.sh my-tag

# Build and push to registry
./docker/build_docker.sh --push

# Custom tag and push
./docker/build_docker.sh my-tag --push
```

The build context is the repo root. A `.dockerignore` at the repo root excludes
`.git/`, build artifacts, and non-essential directories to keep the context small.
Code and `assets/` are baked into the image. Each `assets/` subdirectory is a
separate layer for better caching — code changes don't invalidate asset layers.

## Run

```bash
# Interactive shell with display/GUI forwarding (X11, cache + repo mounts)
./docker/run_docker.sh

# Or specify a custom tag
./docker/run_docker.sh my-tag

```

### Running a single command

```bash
docker run --rm \
    --gpus all \
    --network=host \
    --entrypoint /workspace/isaaclab/_isaac_sim/python.sh \
    -e ACCEPT_EULA=Y \
    robolab:<tag> \
    <script.py> [args...]
```

### Running with display (for GUI/rendering)

```bash
docker run --rm -it \
    --gpus all \
    --network=host \
    --entrypoint /bin/bash \
    -e ACCEPT_EULA=Y \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    robolab:<tag>
```

## What's in the image

- **Base**: `nvcr.io/nvidia/isaac-lab:3.0.0-beta2` (Isaac Sim 6.0) by default; select the legacy stacks with `build_docker.sh --isaac50` or `--isaac51`
- **Code**: `robolab/`, `scripts/`, `examples/`, `tests/`
- **Assets**: `assets/` (~6.5GB)
- **Python packages**: Everything in `requirements.txt`, installed via `pip install -e .`
- **System tools**: `htop`, `nvtop`, `tmux`, `vim`, `git-lfs`, `zip`
