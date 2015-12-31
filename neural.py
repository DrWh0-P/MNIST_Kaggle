import lasagne as le
import theano
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

def show(ind):
    a1=train.ix[ind,:]
    a1=a1.drop('label')
    b1=np.reshape(a1,(28,28))
    plt.imshow(b1,cmap = cm.Greys_r)
    plt.show()

# import data
train=pd.read_csv('train.csv').astype('uint8')
X=train.ix[:,1:].as_matrix().reshape(-1,1,28,28)
X=X/np.float32(256)
Y=train.ix[:,0].as_matrix()

#intializing tensor variables
input_var = theano.tensor.tensor4('inputs')
target_var = theano.tensor.ivector('target')

#creating neural net with 2 hiddenlayers
l_in=le.layers.InputLayer(shape=(None,1,28,28),input_var=input_var)
l_in_drop = le.layers.DropoutLayer(l_in, p=0.2)
l_hid=le.layers.DenseLayer(l_in_drop,num_units=800,nonlinearity=le.nonlinearities.rectify,W=le.init.GlorotUniform())
l_hid_drop = le.layers.DropoutLayer(l_hid, p=0.5)
l_hid2=le.layers.DenseLayer(l_hid_drop,num_units=800,nonlinearity=le.nonlinearities.rectify)
l_hid_drop2 = le.layers.DropoutLayer(l_hid2, p=0.2)
l_out=le.layers.DenseLayer(l_hid_drop2,num_units=10,nonlinearity=le.nonlinearities.softmax)


#generator function for batch SGD
def iterate_batches(x,y,bsize):
    indices=np.arange(len(x))
    np.random.shuffle(indices)
    for start in range(0,len(x)-bsize+1,bsize):
        slice1=indices[start:start+bsize]
        yield x[slice1],y[slice1]
'''

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
''' 

#loss and update function
prediction=le.layers.get_output(l_out)
loss=le.objectives.categorical_crossentropy(prediction,target_var)
loss=loss.mean()
acc = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(prediction, axis=1), target_var),dtype=theano.config.floatX)
params=le.layers.get_all_params(l_out,trainable=True)
updates=le.updates.nesterov_momentum(loss,params,learning_rate=0.01,momentum=0.9)

train_fn=theano.function([input_var,target_var],[loss,acc],updates=updates)
num_epochs=200

# back-propagation-training weights
for epoch in range(num_epochs):
    train_er=0
    train_btch=0
    train_acc=0
    for batch in iterate_batches(X,Y,500):
        input,target=batch
        err,ac=train_fn(input,target)
        train_er+=err
        train_acc+=ac
        train_btch+=1
    print ("Epoch {} of {} t".format(epoch+1,num_epochs))
    print("training loss:\t{:.6f}".format(train_er/train_btch))
    print("training accuracy:\t{:.6f}".format(train_acc/train_btch*100))



