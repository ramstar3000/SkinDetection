import numpy as np
import pickle
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt



class cnnn():
    def __init__(self):
        pass


    def nanargmax(self, arr):
        idx = np.nanargmax(arr)
        idxs = np.unravel_index(idx, arr.shape)  # Get index as multi dimension
        return idxs

    def load(self):
        with open('params2.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            params = pickle.load(f)

            return params

    def relu(self, array):
        array[array <= 0] = 0
        return array

    def softmax(self, X):
        out = np.exp(X)
        return out / np.sum(out)

    def convolution(self, image, filt, bias, s=1):
        (n_f, n_c_f, f, _) = filt.shape  # filter dimensions
        n_c, in_dim, _ = image.shape  # image dimensions

        out_dim = int((in_dim - f) / s) + 1  # calculate output dimensions some mad formula ting

        # Dimensions of filter must match channels of input image

        out = np.zeros((n_f, out_dim, out_dim))  # If 8 filters, 8 dimensions(1) provided

        # convolve the filter over every part of the image, adding the bias at each step.
        # Need to look over and adjust
        for curr_f in range(n_f):
            curr_y = out_y = 0
            while curr_y + f <= in_dim:
                curr_x = out_x = 0
                while curr_x + f <= in_dim:
                    # print(curr_f)
                    out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:, curr_y:curr_y + f, curr_x:curr_x + f]) + \
                                                bias[curr_f]
                    curr_x += s  # Slide across
                    out_x += 1  # Move into next location for output
                curr_y += s  # Slide down
                out_y += 1  # Move 1

        return out

    def maxpool(self, image, f=2, s=2):
        # Downsample image using size of f and stride of s

        n_c, h_prev, w_prev = image.shape  # Get old/current dimensions

        h = ((h_prev - f) / s) + 1  # Calculate new dimensions of img
        w = ((w_prev - f) / s) + 1  # Int been moved so beware int((h_prev - f)/s)+1

        # print(image)
        # print(image[0, 0:2, 4:6])  #[channel num, x:x+2 , y:y+2], is a 2x2 grid

        output = np.zeros((n_c, int(h), int(w)))  # Make empty array to be filled

        for i in range(n_c):  # Once for each channel
            # slide maxpool window over each part of the image and assign the max value at each step to the output
            curr_y = out_y = 0  # Initialised
            while curr_y + f <= h_prev:
                curr_x = out_x = 0
                while curr_x + f <= w_prev:  # Only runs till reaches end of line
                    output[i, out_y, out_x] = np.max(image[i, curr_y:curr_y + f, curr_x:curr_x + f])
                    curr_x += s  # Slide across
                    out_x += 1  # Next index in final array
                curr_y += s  # Slide down
                out_y += 1  # Move into next index downwards

        return output

    def predict(self, image, f1, f2, w3, w4, b1, b2, b3, b4, conv_s=1, pool_f=2, pool_s=2):
        # Run an image through the CNN using the parameters

        conv1 = self.convolution(image, f1, b1)
        conv1 = self.relu(conv1)  # Relu check

        conv2 = self.convolution(conv1, f2, b2)
        conv2 = self.relu(conv2)

        pooled = self.maxpool(conv2)
        (nf2, dim2, _) = pooled.shape
        fc = pooled.reshape((nf2 * dim2 * dim2, 1))  # Flattened

        # Now do predicting with ANN

        z = w3.dot(fc) + b3  # first dense layer
        z = self.relu(z)  # [z<=0] = 0 # pass through ReLU non-linearity

        out = w4.dot(z) + b4  # second dense layer
        probs = self.softmax(out)  # predict class probabilities with the softmax activation function

        return np.argmax(probs), np.max(probs)



    def main(self, image):

        im = Image.open(image)


        im = im.resize((28,28), Image.ANTIALIAS)

        im.save(image, dpi=(600, 600))

        frame = np.asarray(im)
        frame = frame.reshape((3,28,28))  #2352
        print(frame.shape)
        print(frame)

        #naaed to process image
        [f1, f2, w3, w4, b1, b2, b3, b4] = self.load()

        x, y = self.predict(frame, f1, f2, w3, w4, b1, b2, b3, b4)

        return x, y


if __name__ == '__main__':
    X = cnnn()
    print(X.main('temp/tempISIC_0024328.jpg'))



