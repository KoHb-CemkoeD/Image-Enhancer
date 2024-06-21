# Image-Enhancer

## Overview

This project aims to develop a software suite in Python that automates image enhancement using various image processing methods. The primary goal is to provide a fast and convenient way to select and process an input image, choose an enhancement method, display the improved image to illustrate the method's effectiveness, and conduct comparative analysis of image quality. Image processing methods including: lenear, convolution, ML-based.


## Theoretical Basis
### Interpolation Methods
- **Linear Interpolation**: A method to find intermediate values within the range of a discrete set of known values. It applies a linear function to change pixel values of the input image.
<p align="center">
g(x, y) = C · f(x, y) + R  <br>
where f(x, y) and g(x, y) are the input and output images respectively, and C and R are the coefficients of the linear transformation.
</p>

- **Bilinear Interpolation**: Extends linear interpolation for functions of two variables, using the four nearest neighboring pixels to calculate the unknown pixel value.
- **Lanczos Filter**: A multidimensional interpolation filter that smooths digital signal values. It is used for high-quality image processing due to its ability to preserve relative sharpness.

### Convolution and Filters
- **Convolution Kernel**: A small matrix used for blurring, sharpening, and edge detection. It works by convoluting the kernel with the input image, where each pixel's new value is a weighted sum of its neighbors.
- **Gaussian Blur**: A convolution-based filter that uses the Gaussian function to smooth the image by reducing high-frequency noise.

### Deep Learning Methods
Convolutional Neural Networks (CNNs): Used for visual image analysis and processing. Key models include:
- **FSRCNN (Fast Super-Resolution Convolutional Neural Network)**: Provides good image quality with a simple and fast architecture.
- **EDSR (Enhanced Deep Residual Networks)**: Produces higher quality images but operates slower than FSRCNN.
- **ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)**: Combines two neural networks, a generator and a discriminator, to produce and validate high-quality images.

### Image Quality Metrics
- **Mean Squared Error (MSE)**: Measures the average squared difference between estimated and actual values.
- **Peak Signal-to-Noise Ratio (PSNR)**: A higher PSNR indicates greater similarity between the restored and original images. This metric is widely used for practical image quality assessment.


## Project Structure

The project consists of two main modules:

1. **Image Processing Module**: Implements various image enhancement algorithms.
2. **User Interface Module**: Provides a graphical interface for users to interact with the application.


## Implementation
### Language and Libraries
- **Python**: Primary programming language for the project.
- **PyTorch**: Used for implementing machine learning algorithms.
- **OpenCV**: Utilized for image interpolation methods.
- **Qt**: Employed for designing the graphical user interface.


### Software Modules
- **ProcessingThread Class**: Contains methods for loading pre-trained neural network models, adjusting processing parameters, and interacting with the user interface.
- **MainWindow Class**: Manages the main application window and handles user actions such as image opening, preview, and saving. It also integrates UI elements created with Qt Designer.


### User Interface
The main form includes:

A button for opening images.
Information about the application.
Settings panel for selecting and adjusting processing methods.
A preview window for assessing results.
After selecting an image, users can adjust the enhancement method parameters and view the results in four preview windows showing different processing techniques (interpolation, convolution, and neural networks).


## Usage
- **Image Selection**: Users can drag and drop an image onto the application or open it via a file dialog.
- **Parameter Adjustment**: Users can fine-tune processing methods and preview results.
- **Result Saving**: The processed image can be saved through the main menu or by pressing Ctrl+S, specifying the save path, file name, and enhancement method.


## Testing and Results
The effectiveness of enhancement methods was tested on various images with different defects and content types. The results indicated that neural network-based methods, particularly ESRGAN, provided the best quality improvement.


## Conclusion
The project successfully developed a software suite for automated image quality enhancement using interpolation methods and deep learning techniques. The application, built with Python, OpenCV, PyTorch, and Qt, offers a user-friendly interface that allows users to efficiently perform target tasks. ESRGAN emerged as the most effective model for this task.
