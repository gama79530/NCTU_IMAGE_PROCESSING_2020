# NCTU_IMAGE_PROCESSING_2020

This repository contains 3 homework projects of the Image Processing course at NCTU CS department in 2020.

## Enviornment (python 3.8)
- python -m venv venv (or python3 -m venv venv)
- activate venv
- pip install -r requirement.txt


## PROJECT_1

Your first task is to enhance the supplied images using techniques of contrast adjustment, noise reduction, color correction, and so on. There are 6 images supplied (5 regular photos, and one CT). They might not need the same kind of processing. As a result, you should write your processing techniques as modules so that you can try different combinations or parameters for different images. Try to do your best.

## PROJECT_2
This project is about image segmentation techniques that make use of the detection of edge points or image gradient information. Choose at least four test images from those supplied for the previous assignment. You can add your own test images as well.

### Task \#1: Edge detection. 
Try the various gradient filters on the images as well as the LoG method and see what happens. Threshold the gradient magnitudes to find the candidate edge points. Try to compare the results, such as how meaningful the edge points are. Also experiment with how preprocessing (smoothing, contrast adjustment, etc.) affects the process of edge detection.

### Task \#2: Choose only ONE of the following:
1. The implementation of Canny edge detector. Study the effect of the two thresholds. Compare the results to those of task #1.
2. The implementation of Hough transform for straight edge detection. You can use the edge points after thresholding the gradient magnitude image as inputs. You can also skip the thresholding and just use the
gradient magnitude as the "weight" for each pixel. Optional here: Can you find a simple method to separate individual line segments that are parts of the same global line?
3. The watershed algorithm for segmentation. Check how smoothing affects the results. Try this on color
images as well. You will need to define "magnitude of gradient" for color images.

## PROJECT_3
The objective of this assignment is for you to experiment with the various components of a JPEG codec.  
These components include:
1. Block based DCT
2. Quantization of DCT coefficients (with adjustable quality)
3. Predictive coding (between the DC coefficients of adjacent blocks)
4. Run-length coding of the AC coefficients
5. Chromatic subsampling
6. Huffman coding