from torch import from_numpy
import numpy as np


def MNIST_DATA(mnist_trainset,mnist_testset):
    trainset = [i for i in mnist_trainset]
    testset = [i for i in mnist_testset]

    x_train = []
    y_train = []
    
    #Extracting and Reformulating data
    for i in range(len(trainset)):
        x,y = trainset[i][0],trainset[i][1]
        x_train.append(np.asarray(x))
        y_train.append(np.asarray(y))
    for i in range(len(testset)):
        x,y = testset[i][0],testset[i][1]
        x_train.append(np.asarray(x))
        y_train.append(np.asarray(y))

    #Converting list to numpy array
    x_train = np.asarray(x_train)[..., None]
    y_train = np.asarray(y_train)
    print('========= Converting list to numpy array =========')
    print(f'Dataset Shape: {x_train.shape}')
    print(f'Dataset Shape: {y_train.shape}')
    print('')
          
    #Transforming Data into Labels
    X_mean = x_train.mean(axis=(1,2,3))[:,None,None,None]
    X_std = x_train.std(axis=(1,2,3))[:,None,None,None]
    X_norm = (x_train-X_mean)/X_std
    thresh = .5
    Y = X_norm >= thresh
    imgs = []
    lbls = []
    rest = []
    rest_imgs = []
    for i in range(0,10):
        num_masked = i
        Y_num = Y * (y_train==num_masked)[:, None, None, None]
        for i in range(Y_num.shape[0]):
            if True in Y_num[i]:
                lbls.append(Y_num[i])
                imgs.append(x_train[i])
            if not True in Y_num[i]:
                rest.append(Y_num[i])
                rest_imgs.append(x_train[i])

    print('========= Transforming Data into Labels =========')              
    print(f'Discarded Labels: {len(rest)}')
    print(f'Discarded Images: {len(rest_imgs)}')
    print(f'Useful Labels: {len(lbls)}')
    print(f'Useful Images: {len(imgs)}')
    print('')
  
    #Transforming Data into Numpy array
    Y_num = np.asarray(lbls)
    X = np.asarray(imgs)

    X_true = X
    Y_true = Y_num

    print('========= Transforming Data into Numpy array =========')
    print(f'Input Shape: {X_true.shape}')
    print(f'Labels Shape:{Y_true.shape}')
    print('')

    return X_true, Y_true


def TENSOR_DATA(x_train,x_test,y_train,y_test,images_torch_pred,labels_torch_pred):
    #Removing a channel
    x_train = x_train[...,0]
    x_test = x_test[...,0]
    y_train = y_train[...,0]
    y_test = y_test[...,0]
    images_torch_pred = images_torch_pred[...,0]
    labels_torch_pred = labels_torch_pred[...,0]

    #BHWC to BCHW
    x_train = x_train[:,None,:,:]
    x_test = x_test[:,None,:,:]
    y_train = y_train[:,None,:,:]
    y_test = y_test[:,None,:,:]
    images_torch_pred = images_torch_pred[:,None,:,:]
    labels_torch_pred = labels_torch_pred[:,None,:,:]

    # Converting to float32
    x_train = np.float32(x_train)
    x_test = np.float32(x_test)
    y_train = np.float32(y_train)
    y_test = np.float32(y_test)
    images_torch_pred = np.float32(images_torch_pred)
    labels_torch_pred = np.float32(labels_torch_pred)

    print(f'Type of the data. Must be float32: {x_train.dtype}')

    #Transforming to Tensor
    x_train = from_numpy(x_train)
    x_test = from_numpy(x_test)
    y_train = from_numpy(y_train)
    y_test = from_numpy(y_test)
    images_torch_pred = from_numpy(images_torch_pred)
    labels_torch_pred = from_numpy(labels_torch_pred)
    print('============================================================')
    print(f'X_train shape: {x_train.shape} and format: {type(x_train)}')
    print(f'Y_train shape: {y_train.shape} and format: {type(y_train)}')
    print('')
    print('============================================================')
    print(f'X_test shape: {x_test.shape} and format: {type(x_test)}')
    print(f'Y_test shape: {y_test.shape} and format: {type(y_test)}')
    print('')
    print('============================================================')
    print(f'Pred images shape: {images_torch_pred.shape} and format: {type(images_torch_pred)}')
    print(f'Pred labels shape: {labels_torch_pred.shape} and format: {type(labels_torch_pred)}')


    return x_train,x_test,y_train,y_test, images_torch_pred, labels_torch_pred
