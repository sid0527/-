import SimpleITK as sitk
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu
import scipy.ndimage as ndimage


def read_image(path):
    itkimage = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(itkimage)
    return image


def out_image(image, image_name='new_img.jpg'):
    image = (image * 255.0).astype('uint8') 
    image_save = Image.fromarray(image) 
    imageio.imsave(image_name, image_save)


def normalize_image(image):
    image = image.astype(np.float64)
    pixel_min = np.min(image)
    pixel_max = np.max(image)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            for k in range(0, image.shape[2]):
                image[i][j][k] = (image[i][j][k] - pixel_min) / (pixel_max - pixel_min)
    return image


def erosion_dilation_3d(image0):
    # image1 = ndimage.binary_erosion(image0, structure=[[[False,False,True,False,False],
    #                                                     [False,True,True,True,False],
    #                                                     [True,True,True,True,True],
    #                                                     [False,True,True,True,False],
    #                                                     [False,False,True,False,False]],
    #                                                     [[False,True,True,True,False],
    #                                                     [False,True,True,True,False],
    #                                                     [True,True,True,True,True],
    #                                                     [False,True,True,True,False],
    #                                                     [False,True,True,True,False]],
    #                                                     [[True,True,True,True,True],
    #                                                     [True,True,True,True,True],
    #                                                     [True,True,True,True,True],
    #                                                     [True,True,True,True,True],
    #                                                     [True,True,True,True,True]],
    #                                                     [[False,True,True,True,False],
    #                                                     [False,True,True,True,False],
    #                                                     [True,True,True,True,True],
    #                                                     [False,True,True,True,False],
    #                                                     [False,True,True,True,False]],
    #                                                     [[False,False,True,False,False],
    #                                                     [False,True,True,True,False],
    #                                                     [True,True,True,True,True],
    #                                                     [False,True,True,True,False],
    #                                                     [False,False,True,False,False]]])
    # out_image(image1[85, :, :], '1010_imageROI_85_binary_erosion.jpg')
    # image2 = ndimage.binary_dilation(image1, structure=[[[False,False,True,False,False],
    #                                                     [False,True,True,True,False],
    #                                                     [True,True,True,True,True],
    #                                                     [False,True,True,True,False],
    #                                                     [False,False,True,False,False]],
    #                                                     [[False,True,True,True,False],
    #                                                     [False,True,True,True,False],
    #                                                     [True,True,True,True,True],
    #                                                     [False,True,True,True,False],
    #                                                     [False,True,True,True,False]],
    #                                                     [[True,True,True,True,True],
    #                                                     [True,True,True,True,True],
    #                                                     [True,True,True,True,True],
    #                                                     [True,True,True,True,True],
    #                                                     [True,True,True,True,True]],
    #                                                     [[False,True,True,True,False],
    #                                                     [False,True,True,True,False],
    #                                                     [True,True,True,True,True],
    #                                                     [False,True,True,True,False],
    #                                                     [False,True,True,True,False]],
    #                                                     [[False,False,True,False,False],
    #                                                     [False,True,True,True,False],
    #                                                     [True,True,True,True,True],
    #                                                     [False,True,True,True,False],
    #                                                     [False,False,True,False,False]]])
    # out_image(image2[85, :, :], '1010_imageROI_85_binary_dilation.jpg')
    image1 = ndimage.binary_erosion(image0, structure=[[[False,True,False],
                                                        [True,True,True],
                                                        [False,True,False]],
                                                        [[True,True,True],
                                                        [True,True,True],
                                                        [True,True,True]],
                                                        [[False,True,False],
                                                        [True,True,True],
                                                        [False,True,False]]])
    out_image(image1[85, :, :], '1010_imageROI_85_binary_erosion.jpg')
    image2 = ndimage.binary_dilation(image1, structure=[[[False,True,False],
                                                        [True,True,True],
                                                        [False,True,False]],
                                                        [[True,True,True],
                                                        [True,True,True],
                                                        [True,True,True]],
                                                        [[False,True,False],
                                                        [True,True,True],
                                                        [False,True,False]]])
    out_image(image2[85, :, :], '1010_imageROI_85_binary_dilation.jpg')
    return image2

def get_result_image(image, binary):
    return np.multiply(image, binary)

image_original = read_image(r'./ct_train_1010_imageROI.nii')
image_normal = normalize_image(image_original)
out_image(image_normal[85, :, :], '1010_imageROI_85_origin.jpg')
thresh = threshold_otsu(image_normal) 
image_binary = image_normal > thresh 
out_image(image_binary[85, :, :], '1010_imageROI_85_binary.jpg')
image_binary = erosion_dilation_3d(image_binary)
image_result =get_result_image(image_normal, image_binary)
out_image(image_result[85, :, :], '1010_imageROI_85_result.jpg')
