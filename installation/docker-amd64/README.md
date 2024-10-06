# Installation on AMD64 platforms with Docker
## Instructions to build the environment

**Prerequisites**

* `docker` (`docker --version` >= v20.10). [Install here.](https://docs.docker.com/engine/)
* `docker compose` (`docker compose version` >= v2). [Install here.](https://docs.docker.com/compose/install/)

**Build**

We recommend building on an `AMD64` platform, although the Docker BuildKit allows for cross-platform builds.

All commands should be run from the `installation/docker-amd64` directory.

```bash
cd installation/docker-amd64
```

1. Create an environment file for your personal configuration with
   ```bash
   ./template.sh env
   ```
   This creates a `.env` file with pre-filled values.
    - The `USRID` and `GRPID` are used to give the container user read/write access to the volumes that will be mounted
      when the container is run, containing the code of the project, the data, and where you'll write your outputs.
      These need to match the user permissions on the mounted volumes.
      (If you're deploying locally, i.e. where you're building, these values should be filled correctly by default.)

      (**EPFL Note:** _These will typically be your GASPAR credentials and will match the permissions on your lab NFS.
      CLAIRE members should use the `claire-storage` group._)
    - You can ignore the rest of the variables after `## For running locally`.
      These don't influence the build, they will be used later to run your image.

2. Build the images with
   ```bash
   ./template.sh build
   ```
   This will build both the `runtime` and `dev` images and tag them with both `latest-*` and the latest git commit hash
   `<sha>-*`. 
   The runtime images will be used to run the code in an unattended way.
   The dev image has additional utilities that facilitate development in the container.

## Instructions to run the environment

We provide the following guides for running the environment:
- To run on the same machine where you built the image, follow
  [Running locally with Docker Compose](#running-locally-with-docker-compose).

  Moreover, if this machine is a remote server, you can also plug the remote development features
  of popular IDEs such as VSCode or PyCharm with the Docker Compose service running the environment.
- To run on the EPFL Run:ai cluster refer to the `./EPFL-runai-setup/README.md`.

  The guide also provides instructions to do remote development on the Run:ai cluster.
  Other managed cluster users can get inspiration from it too, but we leave it to you to deploy on your managed cluster.

### Running locally with Docker Compose

**Prerequisites**

Steps prefixed with [CUDA] are only required to use NVIDIA GPUs.

* `docker` (`docker --version` >= v20.10). [Install here.](https://docs.docker.com/engine/)
* `docker compose` (`docker compose version` >= v2). [Install here.](https://docs.docker.com/compose/install/)
* [CUDA] [Nvidia CUDA Driver](https://www.nvidia.com/download/index.aspx) (Only the driver. No CUDA toolkit, etc)
* [CUDA] `nvidia-docker` (the NVIDIA Container
  Toolkit). [Install here.](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

**Run**

Edit the `.env` file to specify

- whether you want to run on `cpu` or `gpu` with the `CPU_OR_GPU` variable.
- the local directories to mount the project code, data, and outputs.
  These are specified by the `LOCAL_*_DIR` variables.
  By default, they are set to the project directory on your machine.

Then you can:

- Start the development container with
    ```bash
    ./template.sh up
    ```
  This will start a container running the development image in the background.
  It has an entrypoint that installs the project,
  checking that the code directory has correctly been mounted.

  You can check its logs with
    ```bash
    ./template.sh logs
    ```
  and open a shell in this background container with
    ```bash
    ./template.sh shell
    ```
  You can stop the container or delete it with
    ```bash
    # To stop.
    ./template.sh stop
    # Which can then be restarted with
    ./template.sh start 
    # Or, to delete.
    ./template.sh down
    ```

- Run jobs in independent containers running the `runtime` image with
    ```bash
    ./template.sh run your_command
    ./template.sh run python --version
    ./template.sh run python -m a_project.some_experiment some_arg=some_value
    ```
  These containers start with the entrypoint and then run the command you specified.
  By default, they are automatically removed after they exit.
  The not-so-nice syntax is due to `make` which is not really made to be used like this.

You should not need to override the entrypoint of the service container.
It is necessary to install the project from its mounted location.
Only do so, if you need to debug the container, or you have a custom use case.

## Instructions to maintain the environment

System dependencies are managed by both `apt` and `conda`.
Python dependencies are managed by both `conda` and `pip`.

- Use `apt` for system programs (e.g. `sudo`, `zsh`, `gcc`).
- Use `conda` for non-Python dependencies needed to run the project code (e.g. `mkl`, `swig`).
- Use `conda` for Python dependencies packaged with more than just Python code (e.g. `pytorch`, `numpy`).
  These will typically be your main dependencies and will likely not change as your project grows.
- Use `pip` for the rest of the Python dependencies.
- For more complex dependencies that may require a custom installation or build, use the `Dockerfile` directly.

Here are references and reasons to follow the above claims:

* [A guide for managing `conda` + `pip` environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment).
* [Reasons to  use `conda` for not-Python-only dependencies](https://numpy.org/install/#numpy-packages--accelerated-linear-algebra-libraries).
* [Ways of combining `conda` and `pip`](https://towardsdatascience.com/conda-essential-concepts-and-tricks-e478ed53b5b#42cb).

There are two ways to add dependencies to the environment:

1. **Manually edit the dependencies files.**
   This will be needed the first time you set up the environment.
   It will also be useful if you run into conflicts and have to restart from scratch.
2. **Add/upgrade dependencies interactively** while running a shell in the container to experiment with which
   dependency is needed.
   This is probably what you'll be doing after building the image for the first time.

In both cases, after any change, a snapshot of the full environment specification should be written to the dependencies
files.
We describe how to do so in the Freeze the Environment section.

### Manual editing (before/while building)

- To edit the `apt` dependencies, edit the `dependencies/apt-*.txt` files.
  `apt` dependencies are separated into three files to help with multi-stage builds and keep final images small.
    - In `apt-build.txt` put the dependencies needed to build the environment, e.g. compilers, build tools, etc.
      We provide a set of minimal dependencies as an example.
    - In `apt-runtime.txt` put the dependencies needed to run the environment, e.g. image processing libraries, etc.
    - In `apt-dev.txt` put the utilities that will help you develop in the container, e.g. `htop`, `vim`, etc.

  If you're not familiar with which dependencies are needed for each stage, you can start with the minimal set we
  give and when you encounter errors during the image build, add the missing dependencies to the stage where the error
  occurred.
- To edit the `conda` and `pip` dependencies, edit the `dependencies/environment.yml` file.
- To edit the more complex dependencies, edit the `Dockerfile`.

When manually editing the dependencies files, you do not need to specify the specific version of the dependencies.
These will be written to the environment files when you freeze the environment.
You can of course specify the major versions of specific dependencies you need.

### Interactively (while developing)

`conda` dependencies should all be installed before any `pip` dependency.
This will cause conflicts otherwise as `conda` doesn't track the `pip` dependencies.
So if you need to add a `conda` dependency after you already installed some `pip` dependencies, you need to recreate
the environment by manually adding the dependencies before the build as described in the previous section.

* To add `apt`  dependencies run `sudo apt-install install <package>`
* To add `conda` dependencies run `(conda | pip) install <package>`

### Freeze the environment

After any change to the dependencies, a snapshot of the full environment specification should be written to the
dependencies files.
This includes changes during a build and changes made interactively.
This is to ensure that the environment is reproducible and that the dependencies are tracked at any point in time.

To do so, run the following from the login shell in the container.
The script overwrites the `dependencies/environment.yml` file with the current environment specification,
so it's a good idea to commit the changes to the environment file before/after running it.

```bash
update_env_file
```

For `apt` dependencies add them manually to the `apt-*.txt` files.

For dependencies that require a custom installation or build, edit the `Dockerfile`.

## Troubleshooting

### My image doesn't build with my initial dependencies.

Try removing the dependencies causing the issue, rebuilding, and then installing them interactively when running the
container.
The error messages will possibly be more informative and you will be able to dig into the issue.

Alternatively, you can open a container at the sub-image before the installation of the conda environment, say at
`apt-build-base`, and try to install the conda environment manually.

## Acknowledgements

This Docker setup is heavily based on the [Cresset template](https://github.com/cresset-template/cresset).
We thank them for their work and for making it available to the community.
