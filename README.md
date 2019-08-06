# TCKernelChallenge
The code is a response to this online challenge: https://www.topcoder.com/challenges/30094076

[ImgtoNpArray.py](https://github.com/Zhiwei-Z/TCKernelChallenge/blob/master/ImgToNPArray.ipynb]): process each single kernel image into a numpy array and save it into test_imgarr folder


[NaiveNN.py](https://github.com/Zhiwei-Z/TCKernelChallenge/blob/master/NaiveNN.ipynb): Implement a naive neural network, and SVM to classify kernels.
    1. Naive neural network only performs a bit better than random guessing
    2. SVM has high training accuracy but poor validation accuracy. One possible reason is that there are too many irrelavent pixels in
      and image; possible solution: perform PCA on the image
      
[image_processing folder](https://github.com/Zhiwei-Z/TCKernelChallenge/tree/master/image_processing): contains code that preprocess each image:
  since multiple kernels are taken at once, we need to write pograms that approximately crop out each kernel in a certain order, and label them in the 
  same order
