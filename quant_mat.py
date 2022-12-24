import numpy as np
'''
A few quantization matrices are defined below:
ijg : the standard quantization table for jpeg from the 'Independent JPEG Group' - More info: https://www.ipol.im/pub/art/2022/399/article_lr.pdf
low_comp : quantization table with low compression performance
high_comp : quantization table with high compression performance
lum and chrom : based http://www.impulseadventure.com/photo/jpeg-quantization.html 
'''
def load_quantization_matrix (mode = 'ijg'):
  if mode == 'ijg' or mode == None:
    q = np.array([[16,11,10,16,24,40,51,61],
                  [12,12,14,19,26,58,60,55],
                  [14,13,16,24,40,57,69,56 ],
                  [14,17,22,29,51,87,80,62],
                  [18,22,37,56,68,109,103,77],
                  [24,35,55,64,81,104,113,92],
                  [49,64,78,87,103,121,120,101],
                  [72,92,95,98,112,100,103,99]])
  elif mode == 'low_comp':
    q = np.array([[1, 1, 1, 1, 1, 2, 2, 4],
                  [1, 1, 1, 1, 1, 2, 2, 4],
                  [1, 1, 1, 1, 2, 2, 2, 4],
                  [1, 1, 1, 1, 2, 2, 4, 8],
                  [1, 1, 2, 2, 2, 2, 4, 8],
                  [2, 2, 2, 2, 2, 4, 8, 8],
                  [2, 2, 2, 4, 4, 8, 8, 16],
                  [4, 4, 4, 4, 8, 8, 16, 16]])
  elif mode == 'high_comp':
    q = np.array([[1, 2, 4, 8, 16, 32, 64, 128],
                  [2, 4, 4, 8, 16, 32, 64, 128],
                  [4, 4, 8, 16, 32, 64, 128, 128],
                  [8, 8, 16, 32, 64, 128, 128, 256],
                  [16, 16, 32, 64, 128, 128, 256, 256],
                  [32, 32, 64, 128, 128, 256, 256, 256],
                  [64, 64, 128, 128, 256, 256, 256, 256],
                  [128, 128, 128, 256, 256, 256, 256, 256]])
  elif mode == 'lum':
    q = np.array([[2, 2, 2, 2, 3, 4, 5, 6],
              [2, 2, 2, 2, 3, 4, 5, 6],
              [2, 2, 2, 2, 4, 5, 7, 9],
              [2, 2, 2, 4, 5, 7, 9, 12],
              [3, 3, 4, 5, 8, 10, 12, 12],
              [4, 4, 5, 7, 10, 12, 12, 12],
              [5, 5, 7, 9, 12, 12, 12, 12],
              [6, 6, 9, 12, 12, 12, 12, 12]])
  elif mode == 'chrom':
    q = np.array([[3, 3, 5, 9, 13, 15, 15, 15],
              [3, 4, 6, 11, 14, 12, 12, 12],
              [5, 6, 9, 14, 12, 12, 12, 12],
              [9, 11, 14, 12, 12, 12, 12, 12],
              [13, 14, 12, 12, 12, 12, 12, 12],
              [15, 12, 12, 12, 12, 12, 12, 12],
              [15, 12, 12, 12, 12, 12, 12, 12],
              [15, 12, 12, 12, 12, 12, 12, 12]])
  else: print ('mode is not found!')
  return q