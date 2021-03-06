"""
Definition of image-specific transform classes
"""

# pylint: disable=too-few-public-methods

import numpy as np


class RescaleTransform:
    """Transform class to rescale images to a given range"""
    def __init__(self, out_range=(0, 1), in_range=(0, 255)):
        """
        :param out_range: Value range to which images should be rescaled to
        :param in_range: Old value range of the images
            e.g. (0, 255) for images with raw pixel values
        """
        self.min = out_range[0]
        self.max = out_range[1]
        self._data_min = in_range[0]
        self._data_max = in_range[1]

    def __call__(self, images):
        ########################################################################
        # TODO:                                                                #
        # Rescale the given images:                                            #
        #   - from (self._data_min, self._data_max)                            #
        #   - to (self.min, self.max)                                          #
        ########################################################################

        for img in images:
            for i, channel in enumerate(img):
                for j, p in enumerate(channel):
                    rescaled_pixel = self.rescale_interval(
                        p, 0, 255, self.min, self.max)

                    img[i][j] = rescaled_pixel
        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return images
    
    def rescale_interval(self, value, value_min, value_max, a, b):
        return (b - a) * ((value - value_min) / (value_max - value_min)) + a
        
    

def compute_image_mean_and_std(images):
    """
    Calculate the per-channel image mean and standard deviation of given images
    :param images: numpy array of shape NxHxWxC
        (for N images with C channels of spatial size HxW)
    :returns: per-channels mean and std; numpy array of shape C
    """
    mean, std = None, None
    ########################################################################
    # TODO:                                                                #
    # Calculate the per-channel mean and standard deviation of the images  #
    # Hint: You can use numpy to calculate the mean and standard deviation #
    ########################################################################
    means, stds = [], []
    
    channel_A = []
    channel_B = []
    channel_C = []
    
    for img in images:
        n_channels = img.ndim
        curr_mean = np.mean(img, axis=tuple(range(n_channels-1)))
        curr_std = np.std(img, axis=tuple(range(n_channels-1)))
        
        channel_A.append(img[0])
        channel_B.append(img[1])
        channel_C.append(img[2])
        
        means.append(curr_mean)
        stds.append(curr_std)
        
    mean = [np.mean(channel_A), np.mean(channel_B), np.mean(channel_C)]
    std = [np.std(channel_A), np.std(channel_B), np.std(channel_C)]
        

    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return mean, std


class NormalizeTransform:
    """
    Transform class to normalize images using mean and std
    Functionality depends on the mean and std provided in __init__():
        - if mean and std are single values, normalize the entire image
        - if mean and std are numpy arrays of size C for C image channels,
            then normalize each image channel separately
    """
    def __init__(self, mean, std):
        """
        :param mean: mean of images to be normalized
            can be a single value, or a numpy array of size C
        :param std: standard deviation of images to be normalized
             can be a single value or a numpy array of size C
        """
        self.mean = mean
        self.std = std

    def __call__(self, images):
        ########################################################################
        # TODO:                                                                #
        # normalize the given images:                                          #
        #   - substract the mean of dataset                                    #
        #   - divide by standard deviation                                     #
        ########################################################################

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return images


class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""
    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, images):
        for transform in self.transforms:
            images = transform(images)
        return images
