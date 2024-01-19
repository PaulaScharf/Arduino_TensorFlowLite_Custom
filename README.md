# TensorFlow Lite Micro Library for Arduino

This repository has the code (including examples) needed to use Tensorflow Lite Micro on an Arduino.

## Table of contents
<!--ts-->
* [Table of contents](#table-of-contents)
* [How to Install](#how-to-install)
  * [GitHub](#github)
  * [Checking your Installation](#checking-your-installation)
* [Compatibility](#compatibility)
* [License](#license)
* [Contributing](#contributing)
<!--te-->

## How to Install

### GitHub

The officially supported TensorFlow Lite Micro library for Arduino resides
in the [tflite-micro-arduino-examples](https://github.com/PaulaScharf/Arduino_TensorFlowLite)
GitHub repository.
To install the in-development version of this library, you can use the
latest version directly from the GitHub repository. This requires you clone the
repo into the folder that holds libraries for the Arduino IDE. The location for
this folder varies by operating system, but typically it's in
`~/Arduino/libraries` on Linux, `~/Documents/Arduino/libraries/` on MacOS, and
`My Documents\Arduino\Libraries` on Windows.

Once you're in that folder in the terminal, you can then grab the code using the
git command line tool:

```
git clone git@github.com:PaulaScharf/Arduino_TensorFlowLite.git Arduino_TensorFlowLite
```

To update your clone of the repository to the latest code, use the following terminal commands:
```
cd Arduino_TensorFlowLite
git pull
```

### Checking your Installation

Once the library has been installed, you should then start the Arduino IDE.
You will now see an `Arduino_TensorFlowLite`
entry in the `File -> Examples` menu of the Arduino IDE. This submenu contains a list
of sample projects you can try out.

![Hello World](docs/hello_world_screenshot.png)

## Compatibility

This library was originally designed for the `Arduino Nano 33 BLE Sense` board. I changed that tho.

## License

This code is made available under the Apache 2 license.

## Contributing

Forks of this library are welcome and encouraged. If you have bug reports or
fixes to contribute, the source of this code is at [https://github.com/tensorflow/tflite-micro](http://github.com/tensorflow/tflite-micro)
and all issues and pull requests should be directed there.

The code here is created through an automatic project generation process
and may differ from
that source of truth, since it's cross-platform and needs to be modified to
work within the Arduino IDE.
