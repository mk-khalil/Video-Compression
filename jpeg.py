import numpy as np
from math import sqrt, pi, cos, ceil
from quant_mat import *

def dct(block):
    '''
    A function that applies discrete cosine transformation to a block of image b of size nxn
    for an input block of size nxn the output is the nxn dct transformation
    '''
    n = block.shape[0]
    dct_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0:
                dct_mat[i,j] = np.sqrt(1/n)
            else:
                dct_mat[i,j] = np.sqrt(2/n) * cos(((2*j+1)*i*pi)/(2*n))
    temp = np.dot(dct_mat, block)
    return np.dot(temp, dct_mat.T)

def idct(dct_block):
    '''
    A function that applies inverse dct operation based on the dct function implemented
    above. The input is the nxn transformed dct block and the output is the nxn image block
    '''
    n = dct_block.shape[0]
    idct_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0:
                idct_mat[i,j] = np.sqrt(1/n)
            else:
                idct_mat[i,j] = np.sqrt(2/n) * cos(((2*j+1)*i*pi)/(2*n))
    temp = np.dot(idct_mat.T, dct_block)
    return np.dot(temp, idct_mat)

def zigzag (mat2d):
    '''
    A function that takes a 2-D numpy array (of size nxn) as input and reorders the
    elements in a zigzagfashion and outputs a 1-D vector (of size 1 x n^2)
    '''
    rows = mat2d.shape[0] 
    columns = mat2d.shape[1]
    ordered_vec = np.zeros((rows*columns)) 
    mat_reordered=[[] for i in range(rows+columns-1)] 
    for i in range(rows):
      for j in range(columns):
          sum=i+j
          if(sum%2 ==0):
              #add at beginning
              mat_reordered[sum].insert(0,mat2d[i][j])
          else:
              #add at end of the list
              mat_reordered[sum].append(mat2d[i][j])
    ind = 0          
    for i in mat_reordered:
      for j in i:
          ordered_vec[ind] = j
          ind += 1
    return ordered_vec

def inv_zigzag(vec1d):
    '''
    A function that performs the inverse operation of the zigzag method
    It takes a one dimensional vector and produces a re-ordered square matrix
    using the inverse zigzag method
    '''
    rows = columns= int(sqrt(vec1d.shape[0]))
    mt = np.arange((rows*columns)).reshape((rows,columns))

    ordered_vec = np.zeros((rows*columns), dtype = int) 
    mat_reordered=[[] for i in range(rows+columns-1)]

    for i in range(rows):
      for j in range(columns):
          sum=i+j
          if(sum%2 ==0):

              #add at beginning
              mat_reordered[sum].insert(0,mt[i][j])
          else:

              #add at end of the list
              mat_reordered[sum].append(mt[i][j])
          
    ind = 0          
    for i in mat_reordered:
      for j in i:
          ordered_vec[ind] = j
          ind += 1
    block_correct_order = vec1d[np.argsort(ordered_vec)].reshape((rows,columns)) #orginal block order
    return block_correct_order


def run_length_encoding(vec):
    '''applies run length encoding to the given vector by replacing every sequence of zeros with
    a single zero and how much it is repeated. Returns the encoded vector'''
    
    length = len(vec)
    # init an empty list which will store the encoded sequence
    encoded = []
    # init an index
    i = 0
    while i < length:
        # if the current element in the input vector is zero => we need to perform encoding
        if vec[i] == 0:
            # init the count of zeros to 1
            c = 1
            # increment the index by one to inspect the next element
            i += 1
            # while the new element is zero and i is less than the length of the array
            while i < length and vec[i] == 0:
                # increment the count of zeros
                c += 1
                i += 1
            # store in the encoded array one zero and the count of zeros
            encoded += [0, c]
        else:
            # if the the current element is not zero then just add it to the encoded array
            encoded.append(vec[i])
            i += 1
    return encoded

def run_length_decoding(vec):
    '''applies run length decoding to the given vector by expanding zeros according
    to the number stored next to them. Returns the decoded vector'''
    
    length = len(vec)
    # init an empty list which will store the decoded sequence
    decoded = []
    i = 0
    while i < length:
        # if the current element is zero
        if vec[i] == 0:
            # append to the decoded list n zero where n is the element after the zero
            decoded += [0] * vec[i+1]
            # incremnt i by 2 to skip the zero and the count
            i += 2
        else:
            # otherwise append the current element to the decoded sequence
            decoded.append(vec[i])
            i += 1
    return np.array(decoded)

def pad_img_blocks(img, blocksize = 8):
    '''
    A function that pads the frame so it would be divisble by blocks of size 8x8
    which is necessary for the JPEG compression
    Returns the new matrix and the new dimension
    '''
    h, w = img.shape
    # new matrix dimientions
    h_new, w_new = h,w
    # iterate until the new dimensions are divisble by blocksize (8 in case of JPEG)
    while h_new % blocksize != 0:
        h_new +=1
    while w_new % blocksize != 0:
        w_new +=1
    return np.pad(img, ((0, h_new - h), (0, w_new - w))) , h_new, w_new

def encode_jpeg(img, quant_mode ='ijg'):
    '''
    A function that performs JPEG encoding on the entire frame with the following steps:
      - The frame is divided into blocks of size (n x n) and DCT is applied to each frame
      - Each dct frame is divided by the qunatization matrix
      - The output is reordered using the zigzag function and is appended to the output stream
    The output stream is of size frame_height*frame_width*number of blocks
    '''
    # original image dimensions
    h_org, w_org = img.shape
    # pad the image and return padded dimensions
    padded_img, h_pad, w_pad = pad_img_blocks(img)
    # loading the quantization matrix
    quant_table = load_quantization_matrix(quant_mode)

    # init the matrix for the dct results
    img_dct = np.zeros((h_pad, w_pad))
    for i in range (0, h_pad, 8):
        for j in range(0, w_pad, 8):
            # perform 2D dct of the current image block and store it
            img_dct[i:i+8, j:j+8] = dct(padded_img[i:i+8, j:j+8])

    # rep mat the quantization matrix to the size of the image
    quant_mat = np.tile(quant_table, (h_pad//8, w_pad//8))
    # perform quantization by dividing the dct image by the quant mat and cast to int
    img_dct_quant = (img_dct/quant_mat).astype(int)

    # init the vector for storing the result of zig-zag ordering
    zigzag_vec = np.zeros((h_pad * w_pad), dtype = int)
    v = 0 # index for zigzag
    for i in range (0, h_pad, 8):
        for j in range (0, w_pad, 8):
            # perform zig-zag scan ordering for each quantized dct 8x8 block and store it
            zigzag_vec[v: v+64] = zigzag(img_dct_quant[i:i+8, j:j+8])
            v+=64
    # perform run length encoding for the output vector
    rle_vec = run_length_encoding(zigzag_vec)

    # add header information about image dimensions
    output_vec = np.concatenate(([h_org, w_org, h_pad, w_pad], rle_vec))
    return output_vec

def decode_jpeg(img_vec, quant_mode = 'ijg'):
    '''
    A function that performs JPEG decoding to recover an image frame with the following steps:
      - we will start decoding with the huffman decompressed file contains the decoded bitstream
      - we will perform the inverse of the run length encoding (decoding)
      - then output will be reshaped to groups of 8x8 matrices
      - these matrices will be multiplied by the used quantization table
      - 2D idct will be performed on each 8x8 matrix
      - the matrices will be combined to recover the image
    '''
    # extract header information
    h_org, w_org, h_pad, w_pad = img_vec[:4]
    rle_vec_enc = img_vec[4:]

    #run-length decoding
    img_vec = run_length_decoding(rle_vec_enc)
    # loading the quantization matrix
    quant_table = load_quantization_matrix(quant_mode)

    # matrix for storing the quant dct image after reordering
    img_dct_quant = np.zeros((h_pad, w_pad))
    # perform inverse zigzag reordering
    v = 0 # index for zigzag
    for i in range (0, h_pad, 8):
        for j in range (0, w_pad, 8):
            # perform zig-zag scan ordering for each quantized dct 8x8 block and store it
            img_dct_quant[i:i+8, j:j+8] = inv_zigzag(img_vec[v: v+64])
            v+=64
    # rep mat quantization matrix and multiply by dct image
    quant_mat = np.tile(quant_table, (h_pad//8, w_pad//8))
    # rescale by multiplying by quantization matrix
    img_dct = img_dct_quant * quant_mat
    
    # init matrix for storing the recovered image
    img_rec = np.zeros((h_pad, w_pad))
    for i in range(0, h_pad, 8):
        for j in range (0, w_pad, 8):
            # idct for each 8x8 block and store it to reconstruct the image
            img_rec[i:i+8, j:j+8] = idct(img_dct[i:i+8, j:j+8])
    # return the recovered image after trimming to its original shape
    return img_rec[:h_org, :w_org]
