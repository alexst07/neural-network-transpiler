# Neural Network Transpiler
Convert a model from tflite to a C++ source code using Android Neural Network API.

## Build
In a project directory create a build directory
```
$ mkdir build
$ cd build
```

In build directory call cmake to generate the make file, and call the make program
```
$ cmake ..
$ make
```

To verify if the compilation was successfully completed
```
$ .\nnt -h
```

## How to use
```
  -h [ --help ]             Help screen
  -i [ --info ]             Info about model
  -d [ --dot ] arg          Generate dot file
  -m [ --model ] arg        flatbuffer neural network model
  -p [ --path ] arg         store generated files on this path
  -j [ --javapackage ] arg  java package for JNI
```

In all examples, consider I have a mobilenet_quant_v1_224.tflite model file in build directory, the same directory from where I am executing the nnt executaeble.

### Model info
```
./nnt -m mobilenet_quant_v1_224.tflite -i
```
It generate the output:
```
::Inputs::
 Placeholder<UINT8> [1, 224, 224, 3] (quantized)
   └─ Quant: {min:[0], max:[1], scale: [0.00392157], zero_point:[0]}

::Outputs::
 MobilenetV1/Predictions/Softmax<UINT8> [1, 1001] (quantized)
```

### Generating dot file
```
./nnt -m mobilenet_quant_v1_224.tflite -d mobnet.dot
```
The file mobnet.dot was generated on the same directory

### Generating NNAPI files to use on Android
```
./nnt -m mobilenet_quant_v1_224.tflite -j com.nnt.nnexample -p mobnet_path
```
It creates a directory with name "mobnet_path" with files: [jni.cc, nn.h, nn.cc, weights_biases.bin]
where the java package is com.nnt.nnexample
