# C++ PyTorch Deployment

This tutorial demonstrates how to train a Neural Network in Python and deploy the model in C++

![Torch C++ Deployment](https://github.com/ZachWolpe/cpp-torch-deployment/raw/main/torch-cpp.png)

### Overview

1. Train (or download) a PyTorch model.
2. Checkpoint & save the model.
3. Convert the model to TorchScript (a serialisation independent of the Python interpreter).
4. Install and link LibTorch to your C++ application.
5. Perform inference!


## 1. Train & save a model with PyTorch (Optional)

First we'll train a model with PyTorch and track/save it with MLflow. Alternatively, download a pre-trained model.

#### 1.1 Download the training data

Setup the file structure as seen in the repo. Navigate to the `exe_1_build_torch_model` directory.

Download the training data:

```bash
python exe_1_download_data.py
```

#### 1.2 Train the model

We're using MLflow to track & save the model, and checkpointing to resume training.

Launch a training job by running:

```bash
python exe_2_launch_training.py
```

By default, this will resume training from the last checkpoint. To start a new training job, delete the `./mlruns` directory. After the training job a new model (or version) will be saved in the `./models` directory.

`exe_2_launch_training.py` takes a number of command line arguments, run `python exe_2_launch_training.py --help` to see them. 


Optional: To examine the training job, start the MLflow server:

```bash
mlflow ui
```

Finally, before loading the model in C++, we need to convert the model to `TorchScript`. This removes the need for the Python interpreter when loading the model in `C++`.

Pass the desired model path to `exe_4_convert_model_to_torchscript.py` as a command line argument.  


```bash
python exe_4_convert_model_to_torchscript.py --model_path ./models/model.pth
```

Resources:

- [TorchScript to C++](https://pytorch.org/tutorials/advanced/cpp_export.html)
- [TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)



----
## 2. Execute Torchlib

Before trying to port the model make inference call in C++, ensure `torchlib` is working correctly.

Navigate to `exe_2_cpp_torchlib`, 

Download `libtorch` from the [PyTorch website](https://pytorch.org/get-started/locally/).

Create & navigate to a `build` directory:

```bash
mkdir build
cd build
```

Run CMake:

```bash
 cmake -S ../ -B . -DTorch_DIR<absolute-path-to-libtorch>
```

`-S ../` specifies the source directory, `-B .` specifies the build directory, and `Torch_DIR` specifies the path to the `libtorch` directory.


Run `make` to build the project:

```bash
make
```

Your `build` directory should now contain an executable called `torch-cpp-deployment`.

Run the executable:

```bash
./torch-cpp-deployment
```

**MacOS Security Issue**

If you're running MacOS, you may encounter a security issue when running the executable. To resolve this, run the following command:

```bash
// xattr -r -d com.apple.quarantine <path-to-libtorch>
```

This will remove the quarantine attribute from the `libtorch` directory.

If these runs error free, you are ready to port the model to C++!


----
## 3. Port the model to C++

Similar to the previous step, we'll use CMake to build the C++ deployment.

Navigate to the `exe_3_cpp_torch_deplyment` directory.

Setup the build directory.

    
```bash
mkdir build
cd build
```

Configure the `CMakeList.txt` file.

```bash
cmake -S ../ -B . -DTorch_DIR=<Path-to-Libtorch>
```

Build the project:

```bash
make
```

The executable will be called `cpp-torch-deployment`. It takes a single argument, the path to the model file. Our C++ application will now use the model binary to perform inference on a tensor. The tensor is a simple 3-D matrix of 1's (kept simple for demonstative purposes).

```bash
./cpp-torch-deployment <Path-to-torchscript-model.pth>
```

Voila! We have now deployed our model in a `C++` runtime.

