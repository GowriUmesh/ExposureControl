# ExposureControl
Implementing exposure control by maximizing a robust gradient-based image quality metrics to improve robustness of visual Odometry in challenging illumination conditions using HDR images.

This README explains the working of the code prepared by Group 20 for active exposure control for robust visual odometry.
The program uses OpenCV and is written in C++. It implements the control algorithm by optimising the image quality metrics.
The program uses a hardcoded datase. The optimised exposure time for the next frame is calculated based on the dataset and is not augmented.
The program calculates two different gradient based image quality metrics and verifies the best frame by extracting fast features.
The optimisation algorithm is the gradient-ascent method.
The photometric response calculation needs a minimum of 10 frames failing which the response will not be calculated or will be inaccurate.
A change in image format must be altered in the code as the dataset is hardcoded.
The program only accepts images of the resolution 512x768 pixels.
