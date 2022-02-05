import os, h5py, glob, re, argparse
import numpy as  np

###############################################################################
# Utility functions: rect_regression, sigmoid, conv2, conv_layer
def rect_regression(rect, anchor):
    '''Convert a rectangle into a regression target, with respect to  a given anchor rect'''
    return(np.array([(rect[0]-anchor[0])/anchor[2],(rect[1]-anchor[1])/anchor[3],
                     np.log(rect[2]/anchor[2]),np.log(rect[3]/anchor[3])]))

def rect(reg, anchor):
    return(np.array([(reg[0]*anchor[2])+anchor[0],(reg[1]*anchor[3])+anchor[1],np.exp(reg[2])*anchor[2],np.exp(reg[3])*anchor[3]]))

def sigmoid(Xi):
    '''
    Compute forward-pass of a sigmoid layer.

    Input:
      Xi (Ntoks, N1, N2, ND) - excitation
    Output:
      H (Ntoks, N1, N2, ND) - sigmoid activation

    This is provided as a utility function for you, because of weird underflow effects:
    np.exp(-x) generates NaN if x<-100 or so.
    In order to get around that problem, this function just leaves activation=0
    if excitation <= -100.  Feel free to use this, to avoid NaNs.
    '''
    H = np.zeros(Xi.shape)
    H[Xi > -100] = 1/(1+np.exp(-Xi[Xi > -100]))
    return(H)

safe_log_min = np.exp(-100)
def safe_log(X):
    '''
    Compute safe logarithm.
    Input:  X = any numpy array
    Output: 
      Y[X > np.exp(-100)] = np.log(X[X > np.exp(-100)])
      Y = 0 otherwise
    '''
    Y = np.zeros(X.shape)
    Y[X > safe_log_min] = np.log(X[X > safe_log_min])
    return Y

def conv2(H, W, padding):
    '''
    Compute a 2D convolution.  Compute only the valid outputs, after padding H with 
    specified number of rows and columns before and after.

    Input:
      H (N1,N2) - input image
      W (M1,M2) - impulse response, indexed from -M1//2 <= 
      padding (scalar) - number of rows and columns of zeros to pad before and after H

    Output:
      Xi  (N1-M1+1+2*padding,N2-M2+1+2*padding) - output image
         The output image is computed using the equivalent of 'valid' mode,
         after padding H  with "padding" rows and columns of zeros before and after the image.

    Xi[n1,n2] = sum_m1 sum_m2 W[m1,m2] * H[n1-m1,n2-m2]
    
    '''
    N1,N2 = H.shape
    M1,M2 = W.shape
    H_padded = np.zeros((N1+2*padding,N2+2*padding))
    H_padded[padding:N1+padding,padding:N2+padding] = H
    W_flipped_and_flattened = W[::-1,::-1].flatten()
    Xi = np.empty((N1-M1+1+2*padding,N1-M1+1+2*padding))
    for n1 in range(N1-M1+1+2*padding):
        for n2 in range(N2-M2+1+2*padding):
            Xi[n1,n2] = np.inner(W_flipped_and_flattened, H_padded[n1:n1+M1,n2:n2+M2].flatten())
    return Xi

def conv_layer(H, W, padding):
    '''
    Compute a convolutional layer between input activations H and weights W.

    Input:
      H (N1,N2,NC) - hidden-layer activation from the previous layer
      W (M1,M2,NC,ND) - convolution weights
      padding (scalar) - number of rows and columns of zeros to pad before and after H

    Output:
      Xi (Ntoks,N1,N2,ND) - excitations of the next layer
    
    Xi[:,:,d] = sum_c conv2(H[:,:,c], W[:,:,c,d])
    
    This is provided as a utility function for you, because writing it would
    be too tedious.  Feel free to write your own version if you wish.
    '''
    N1, N2, NC = H.shape
    M1, M2, NC, ND = W.shape
    Xi = np.zeros((N1-M1+1+2*padding, N2-M2+1+2*padding, ND))
    for d in range(ND):
        for c in range(NC):
            Xi[:,:,d] += conv2(H[:,:,c], W[:,:,c,d], padding)
    return Xi



###############################################################################
# TODO: here are the functions that you need to write

def forwardprop(X, W1, W2):
    '''
    Compute forward propagation of the FasterRCNN network.

    Inputs:
      X (N1,N2,NC) - input features
      W1 (M1,M2,NC,ND) -  weight tensor for the first layer
      W2 (1,1,ND,NA,NY) - weight tensor for the second layer

    Outputs:
      H (N1,N2,ND) - hidden layer activations
      Yhat (N1,N2,NA,NY) - outputs

    Interpretation of the outputs:
      Yhat[n1,n2,a,:4] - regression output, (n1,n2) pixel, a'th anchor
      Yhat[n1,n2,a,4] - classfication output, (n1,n2) pixel, a'th anchor
    '''
    D = W2.shape[2]
    A = W2.shape[3]
    K = W2.shape[4]
    N1, N2, NC = X.shape
    M1, M2, NC, ND = W1.shape
    Xi_1 = conv_layer(X, W1, 1)
    H = np.maximum(0, Xi_1)   # 8,8,128
    # W2.shape = 1,1,128,9,5
    # Xi_2.shape = 8,8,9,5
    Xi_2 = np.zeros((N1,N2,A,K))
    Y_hat = np.zeros((N1,N2,A,K))
    
    #Y_1 = matmul(
    
    #print(Xi_2.shape)
    for n1 in range(N1):
        for n2 in range(N2):
            for a in range(A):
                for k in range(K):
                    for d in range(D):
                        Xi_2[n1,n2,a,k] += H[n1,n2,d]*W2[0,0,d,a,k]
     
    Y_hat[:,:,:,4] = sigmoid(Xi_2[:,:,:,4])
    
    for i in range(4):
        Y_hat[:,:,:,i] = Xi_2[:,:,:,i]

    return H, Y_hat
def detect(Yhat, number_to_return, anchors):
    '''
    Input:
      Yhat (N1,N2,NA,NY) - neural net outputs for just one image
      number_to_return (scalar) - the number of rectangles to return
      anchors (N1,N2,NA,NY) - the set of standard anchor rectangles
    Output:
      best_rects (number_to_return,4) - [x,y,w,h] rectangles most likely to contain faces.
      You should find the number_to_return rows, from Yhat,
      with the highest values of Yhat[n1,n2,a,4],
      then convert their corresponding Yhat[n1,n2,a,0:4] 
      from regression targets back into rectangles
      (i.e., reverse the process in rect_regression()).
    '''
    N1,N2,NA,NY = Yhat.shape  # (8,8,9,5)
    best_rects = np.zeros((number_to_return,4))
    Yhat_list = []
    Yhat_flat = Yhat.flatten()
    anchors_flat = anchors.flatten()
    #print(anchors.shape, Yhat.shape)
    reg_list = []
    #print(anchors[0][0][0])
    for i in range(0,len(Yhat_flat),5):
        reg_list.append(Yhat_flat[i])
    #print(len(reg_list))
    #print(anchors[0,2,2,:])
    last_element_list = []
    pair_dict = {}
    
    for n1 in range(N1):
        for n2 in range(N2):
            for na in range(NA):
                #last_element_list.append(Yhat[n1,n2,na,4])
                yhat = Yhat[n1,n2,na,:]
                anc = anchors[n1,n2,na,:]
                
                last_element = yhat[4]
                regression = yhat[0:4]
                pair_dict[last_element] = (regression,anc)
                
    best_pairs = dict(sorted(pair_dict.items(),key=lambda x:x[0], reverse=True)[0:number_to_return])

    index = 0
    for items in best_pairs:
        best_rects[index] = rect(list(pair_dict[items][0]), list(pair_dict[items][1]))
        index += 1
         
    #print(best_rects)
    
    
                
    #print(len(last_element_list))
    
    #sorted_ind = np.argsort(last_element_list)
    #largest_ind = sorted_ind[::-1][:number_to_return]
    #print(largest_ind)
    #rect_ind0,rect_ind1,rect_ind2,rect_ind3 = np.unravel_index(largest_ind[0], (N1,N2,NA,NY))
    #print(anchors[rect_index])
    #for num in range(number_to_return):
    #    reg_temp = []
        #anc_temp = np.zeros((N1,N2,NA,4))
    #    anc_temp = []
    #    rect_ind0,rect_ind1,rect_ind2,rect_ind3 = np.unravel_index(largest_ind[num], (N1,N2,NA,NY))
    #    anc_temp = anchors[rect_ind0,rect_ind1,rect_ind2,0:4]
    #    reg_temp = Yhat[rect_ind0,rect_ind1,rect_ind2,0:4]
            #anc_temp.append(anchors_flat[largest_ind[num]+anc])
    #    best_rects[num] = rect(reg_temp, anc_temp)
    #print(best_rects)    
    return best_rects
    
def loss(Yhat, Y):
    '''
    Compute the two loss terms for the FasterRCNN network, for one image.

    Inputs:
      Yhat (N1,N2,NA,NY) - neural net outputs
      Y (N1,N2,NA,NY) - targets
    Outputs:
      bce_loss (scalar) - 
        binary cross entropy loss of the classification output,
        averaged over all positions in the image, averaged over all anchors 
        at each position.
      mse_loss (scalar) -
        0.5 times the mean-squared-error loss of the regression output,
        averaged over all of the targets (images X positions X  anchors) where
        the classification target is  Y[n1,n2,a,4]==1.  If there are no such targets,
        then mse_loss = 0.
    '''
    N1, N2, NA, NY = Y.shape
    num_mse = 0
    den_mse = 0
    num_bce = 0
    den_bce = 0
    #print(N1,N2,NA,NY)
    mse = Y[:,:,:,0:4] - Yhat[:,:,:,0:4]
    mse_norm = (np.linalg.norm(mse))**2
    #print(mse_norm)
    for n1 in range(N1):
        for n2 in range(N2):
            for na in range(NA):
                num_mse += Y[n1,n2,na,4]*(np.linalg.norm(Y[n1,n2,na,0:4] - Yhat[n1,n2,na,0:4]))**2
                den_mse += Y[n1,n2,na,4]
                num_bce += (Y[n1,n2,na,4]*safe_log(Yhat[n1,n2,na,4])) + (1-Y[n1,n2,na,4])*safe_log(1-Yhat[n1,n2,na,4])
                                                  
    den_bce = N1*N2*NA
                
    L_mse = 0.5*(num_mse/den_mse)
    L_bce = -(num_bce)/den_bce
    #print(L_mse)
    #print(product)            
    #print(mse.shape, product.shape)
    #print(Y_norm.shape)    
    #print(Y[3])
    #print(Y_diff.shape)
    return L_bce, L_mse

def backprop(Y, Yhat, H, W2):
    '''
    Compute back-propagation in the Faster-RCNN network.
    Inputs:
      Y (N1,N2,NA,NY) - training targets
      Yhat (N1,N2,NA,NY) - network outputs
      H (N1,N2,ND) - hidden layer activations
      W2 (1,1,ND,NA,NY) - second-layer weights
    Outputs:
      GradXi1 (N1,N2,ND) - derivative of loss w.r.t. 1st-layer excitations
      GradXi2 (N1,N2,NA,NY) - derivative of loss w.r.t. 2nd-layer excitations
    '''
    N1,N2,ND = H.shape
    NA,NY = Y.shape[2], Y.shape[3]
    GradXi1 = np.zeros((N1,N2,ND))
    #GradXi2 = np.zeros((N1,N2,NA,NY))
    
    #for n1 in range(N1):
    #    for n2 in range(N2):
    GradXi2 = Yhat - Y        
    
    for n1 in range(N1):
        for n2 in range(N2):
            for nd in range(ND):
                if H[n1,n2,nd] > 0:
                    GradXi1[n1,n2,nd] = np.sum(GradXi2[n1,n2,:,:] * W2[0,0,nd,:,:])
                    
    
    #print(GradXi1.shape, GradXi2.shape)
    return GradXi1,GradXi2

def weight_gradient(X, H, GradXi1, GradXi2, M1, M2):
    '''
    Compute weight gradient in the Faster-RCNN network.
    Inputs:
      X (N1,N2,NC) - network inputs
      H (N1,N2,ND) - hidden-layer activations
      GradXi1 (N1,N2,ND) - gradient of loss w.r.t. layer-1 excitations
      GradXi2 (N1,N2,NA,NY) - gradient of loss w.r.t. layer-2 excitations
      M1 - leading dimension of W1
      M2 - second dimension of W1
    Outputs:
      dW1 (M1,M2,NC,ND) - gradient of loss w.r.t. layer-1 weights
      dW2 (1,1,ND,NA,NY) -  gradient of loss w.r.t. layer-2 weights
    '''
    N1,N2,NC = X.shape
    ND = H.shape[2]
    NA, NY = GradXi2.shape[2], GradXi2.shape[3]
    dW1 = np.zeros((M1,M2,NC,ND))
    dW2 = np.zeros((1,1,ND,NA,NY))
    #print(M1,M2)
    
    for nd in range(ND):
        for nc in range(NC):
            for m2 in range(M2):
                for m1 in range(M1):
                    sum = 0
                    for n1 in range(N1):
                        for n2 in range(N2):
                            #if -1<=m1<=1 and -1<=m2<=1:
                            if (n1-(m1-1)>=0 and n2-(m2-1)>=0) and (n1-(m1-1))<N1 and (n2-(m2-1))<N2:
                                sum += GradXi1[n1-(m1-1),n2-(m2-1),nd]*X[n1,n2,nc]

                    dW1[m1,m2,nc,nd] = sum
                    
    for nd in range(ND):
        for na in range(NA):
            for ny in range(NY):
                sum1 = 0
                for n1 in range(N1):
                    for n2 in range(N2):
                        sum1 += GradXi2[n1,n2,na,ny]*H[n1,n2,nd]
                dW2[0,0,nd,na,ny] = sum1
    #print(dW1.shape, dW2.shape)
    return dW1,dW2


def weight_update(W1,W2,dW1,dW2,learning_rate):
    '''
    Input: 
      W1 (M1,M2,NC,ND) = first layer weights
      W2 (1,1,ND,NA,NY) = second layer weights
      dW1 (M1,M2,NC,ND) = first layer weights
      dW2 (1,1,ND,NA,NY) = second layer weights
      learning_rate = scalar learning rate
    Output:
      new_W1 (M1,M2,NC,ND) = first layer weights
      new_W2 (1,1,ND,NA,NY) = second layer weights
    '''
    new_W1 = W1 - learning_rate*dW1
    new_W2 = W2 - learning_rate*dW2
    #print(new_W1.shape, new_W2.shape)
    return new_W1, new_W2


