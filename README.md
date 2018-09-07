## Facial Detection Over IP
The goal of this project is to detect Faces in live IP Camera footage. We use Tensorflow as our back end and build off of the
SSD MobileNet Architecture. Versions using Inception V1/V2 are being explored for higher accuracy.

[Video Exposition](https://youtu.be/-ED7T9X5zcY)

## Motivation
We take in a live video stream from a network connected camera and perform image classification on the frames of this video. This video is then sent over LAN to an embedded device for display purposes (i.e. ARM Processor).

By taking this approach, we can allow users to explore image classification output on devices not powerful enough to perform these computations themselves. Additionally, any smartphone can act as the video source.

## Tech/framework used

<b>Built with</b>
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
- [COCO Dataset](http://cocodataset.org/)

## To Do List
- [ ] Make Install Guide for Linux (CPU Only)
- [X] Make Install Guide for Linux (GPU)
- [ ] Reduce calls to frozen graph
- [ ] Rebuild to utilize Intel MKL-DNN, AVX2, JIT
- [ ] Extract frames from video at system edge
- [ ] Apply Transfer Learning and train using [LFW Database](http://vis-www.cs.umass.edu/lfw/)

## Environment Installation / Setup - TensorFlow CPU

  #### Windows
  - Install [Anaconda](https://www.anaconda.com/distribution/) and add it to `$PATH` if you dont already have it. 
  - Install Tensorflow CPU and all dependencies:
    ```
    conda create --name tf python=3.5
    activate tf
    conda install scipy
    pip install tensorflow
    ```
  - If everything is installed correctly then this test code
    ```Python
    >> import tensorflow as tf
    >> hello = tf.constant('Hello, TensorFlow!')
    >> sess = tf.Session()
    >> print(sess.run(hello))
    ```
    Should return
    ```Python
    Hello, TensorFlow!
    ```
  
  #### Linux
  Coming soon. 

## Environment Installation / Setup - TensorFlow + NVIDIA GPU

  #### Disclaimer
  Setting Up TensorFlow GPU will take a long time, and there are no guarantees. The code will run much better on the GPU Version, however installation of TF-GPU on Windows is tedious at best and maddening at worst. **Use this guide at your own risk.** 
  
  #### Windows
  - Ensure you have [Microsoft C/C++ Build Tools 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48159) installed. If you have Visual Studio 2017 installed, these tools *are not* installed by default. Make sure these are in your `$PATH` as well. 
  - Ensure that your [NVIDIA Base Drivers](http://www.nvidia.com/Download/index.aspx) are up to date. At time of writing, the current driver set is Release 390. Reboot after this step.
  - Install the [NVIDIA Cuda Toolkit](https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64). **This must be version 9.0, version 9.1 is not yet supported.** Add the bin folder and enclosing CUDA folder to `$PATH`.
  - Install [CuDNN](https://developer.nvidia.com/cudnn) **version 7.0** and add that to `$PATH`. 
  - Reboot. If you got all of this installed correctly, the tedious part is over.
  - Install [Anaconda](https://www.anaconda.com/distribution/) and add it to `$PATH` if you dont already have it. 
  - Install Tensorflow GPU and all dependencies:
    ```
    conda create --name tf-gpu python=3.5
    activate tf-gpu
    conda install scipy
    pip install tensorflow-gpu
    ```
  - If everything is installed correctly then this test code
    ```Python
    >> import tensorflow as tf
    >> hello = tf.constant('Hello, TensorFlow!')
    >> sess = tf.Session()
    >> print(sess.run(hello))
    ```
    Should return
    ```Python
    Hello, TensorFlow!
    ```
  - Go take a walk because you're done and you deserve it.
  
  #### Linux
  Coming soon. 
  
## Project Dependencies
A full list of dependencies can be found in [env.yml](https://github.com/njohnsoncpe/facialRecognition/blob/master/env.yml). [Anaconda maintains](https://conda.io/docs/commands/env/conda-env-create.html) that an equivalent enviroment to mine can be built using: 
```
conda env create -f env.yml -n tf-gpu 
```
I have been unable to independenly verifiy this functionality. If anyone finds success with this method, let me know and I will update this to reflect any new information.

## Credits
Inspired by [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

## License
MIT License. Provided AS IS.

MIT © [Noah Johnson](https://njohnsoncpe.github.io)
