from numpy.fft import fft2, ifft2
import numpy as np

image = np.zeros((10,10),dtype=np.int)
block = np.arange(1,7).reshape(2,3)
# print(f'{image} \n')
x = 2
y = 3
image[x:x+block.shape[0], y:y+block.shape[1]] = block

# print(image)


kernel = np.array([[0, 1, 0],
          [1, 1, 1],
          [0, 1, 0]])

# result = np.convolve(image, kernel)  

def convolve3d(img, kernel):
    # calc the size of the array of submatracies
    window_shape = tuple(np.subtract(img.shape, kernel.shape) + 1)

    # alias for the function
    strd = np.lib.stride_tricks.as_strided

    # make an array of submatracies
    submatrices = strd(img,kernel.shape + window_shape,img.strides * 2)

    # sum the submatraces and kernel
    convolved_matrix = np.einsum('hi,hikl->kl', kernel, submatrices)
    #3D
    # convolved_matrix = np.einsum('hij,hijklm->klm', kernel, submatrices)

    return convolved_matrix

res = convolve3d(image, kernel)
# print(res)

#! Second

from numpy.lib.stride_tricks import as_strided

img_arr = np.arange(0, 16).reshape(4, 4) 
img_arr = np.pad(img_arr, pad_width=1, mode='edge')
print(img_arr)
window_shape = (3, 3)
view_shape = tuple(np.subtract(img_arr.shape, window_shape) + 1) + window_shape #cos img_dim - window +1
print(f'\nView shape:{view_shape}')
print(f'Img strides:{img_arr.strides*2}\n')
arr_view = as_strided(img_arr, view_shape, img_arr.strides * 2)
print(arr_view) 
# print(arr_view.shape)
arr_view = arr_view.reshape((-1,) + window_shape)
# print(arr_view)
# print(arr_view.shape)

max_row1 = np.apply_along_axis(max, 2, arr_view) #max from last sub row
max_col = np.apply_along_axis(max, 1, max_row1)  # then max sub cols
max_col.shape = (4, 4)
print(max_col)
