# 服务器端C++预测

C++版本的PaddleInference编译


## 1. 准备环境

### 运行准备
- #### Windows环境，使用`Visual Studio 2019`相关编译器编译。

- #### 使用cuda版本CUDA=11.6, cuDNN=8.5
- #### 需要正常访问Github用来下载相关依赖

### 1.1 添加opencv库

* 首先需要从opencv官网上下载好open-cv相关lib库，也可以自行编译

找到open-cv的库文件，找到以下目录，记录下它的路径。

```
opencv3/
|-- bin
|-- include
|-- lib64
|-- share
```

### 1.2 下载或者编译Paddle预测库

* 有2种方式获取Paddle预测库，下面进行详细介绍。

#### 1.2.1 ~~预测库源码编译(不推荐)~~
* 如果希望获取最新预测库特性，可以从Paddle github上克隆最新代码，源码编译预测库。
* 可以参考[Paddle预测库官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#id16)的说明，从github上获取Paddle代码，然后进行编译，生成最新的预测库。使用git获取代码方法如下。

```shell
git clone https://github.com/PaddlePaddle/Paddle.git
```

* 进入Paddle目录后，使用如下参数编译。

```shell
cmake  .. \
    -DWITH_CONTRIB=OFF \
    -DWITH_MKL=ON \
    -DWITH_MKLDNN=ON  \
    -DWITH_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_INFERENCE_API_TEST=OFF \
    -DON_INFER=ON \
    -DWITH_PYTHON=ON
```

更多编译参数选项可以参考Paddle C++预测库官网：[https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#id16](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#id16)。


* 编译完成之后，可以在`build/paddle_inference_install_dir/`文件下看到生成了以下文件及文件夹。

```
build/paddle_inference_install_dir/
|-- CMakeCache.txt
|-- paddle
|-- third_party
|-- version.txt
```

其中`paddle`就是之后进行C++预测时所需的Paddle库，`version.txt`中包含当前预测库的版本信息。

#### 1.2.2 直接下载安装

* [Paddle预测库官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#id1)上提供了不同cuda版本的Linux预测库，可以在官网查看并选择合适的预测库版本，注意必须选择`develop`版本。


## 2 使用IDE配置运行环境

### 2.1 将模型导出为inference model

* 可以参考[模型导出](../../tools/export_model.py)，导出`inference model`，用于模型预测。得到预测模型后，假设模型文件放在`inference`目录下，则目录结构如下。

```
inference/
|--inference.pdmodel
|--inference.pdiparams
```
**注意**：上述文件中，`inference.pdmodel`文件存储了模型结构信息，`inference.pdiparams`文件存储了模型参数信息。模型目录可以随意设置，但是模型名字不能修改。

### 2.2 配置编译器的CMakeList.txt

* CMakeList.txt是绝大多数C++项目的编译描述文件

其编译器的原始编译参数如下所示：
```shell
OPENCV_DIR=your_opencv_dir
LIB_DIR=your_paddle_inference_dir
CUDA_LIB_DIR=your_cuda_lib_dir
CUDNN_LIB_DIR=your_cudnn_lib_dir
TENSORRT_DIR=your_tensorrt_lib_dir

BUILD_DIR=build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DPADDLE_LIB=${LIB_DIR} \
    -DWITH_MKL=ON \
    -DDEMO_NAME=clas_system \
    -DWITH_GPU=OFF \
    -DWITH_STATIC_LIB=OFF \
    -DWITH_TENSORRT=OFF \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DCUDNN_LIB=${CUDNN_LIB_DIR} \
    -DCUDA_LIB=${CUDA_LIB_DIR} \

make -j
cd ..
```

上述命令中，

* `OPENCV_DIR`为opencv编译安装的地址（本例中为`opencv`文件夹的路径）；

* `LIB_DIR`为下载的Paddle预测库（`paddle_inference`文件夹），或编译生成的Paddle预测库（`build/paddle_inference_install_dir`文件夹）的路径；

* `CUDA_LIB_DIR`为cuda库文件地址，在docker中为`/usr/local/cuda/lib64`；

* `CUDNN_LIB_DIR`为cudnn库文件地址，在docker中为`/usr/lib/x86_64-linux-gnu/`。

* `TENSORRT_DIR`是tensorrt库文件地址，在dokcer中为`/usr/local/TensorRT6-cuda10.0-cudnn7/`，TensorRT需要结合GPU使用。

在IDE中，你需要将这些位置以编译参数缓存进工程中，以下为我个人使用的参数，相关依赖位置 **因人而异**：

```
-G "Visual Studio 16 2019" 
-DWITH_GPU:BOOL=ON 
-DPADDLE_LIB:PATH="C:\Program Files\paddle_inference" 
-DCUDA_LIB:PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64" 
-DCUDNN_LIB:PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDNN8.5\lib" 
-DOPENCV_DIR:PATH="C:\Program Files\opencv"
```


## 3 项目结构
#### 3.1 项目结构介绍

```
  — 程序入口：main.cpp
  — 推理预测：inference.cpp
  — 设置参数：cls_config.cpp
  — 推理底层：cls.cpp
```

如不修改项目底层，你只需要关注Inference.cpp下的
```c++
clsInference::clsInferenceInit(const string& yaml_path)
clsInference::clsInferenceRun(Classifier& classifier, const cv::Mat& img)
```
其中Init 函数读取模型的描述文件，返回一个config类型

再调用此对象的Run 函数，参数为config与cv::MAT 执行推理任务

#### 3.2 执行示例

```c++
#include <include/inference.h>
using namespace cv;
using namespace Inference;
int main() {
    clsInference inferenceInit;
    Mat img = cv::imread(R"(C:\Users\ty\Desktop\MyDesktopDirectory\paddleCPPInferDEMO\inference\pic\20210611_101518_0490.jpg)");
    Classifier clsConfig = inferenceInit.clsInferenceInit(R"(C:\Users\ty\Desktop\MyDesktopDirectory\paddleCPPInferDEMO\inference_attr.yaml)");
    inferenceInit.clsInferenceRun(clsConfig, img);
}
```
