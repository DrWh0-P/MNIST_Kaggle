import lasagne as le
import theano
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pickle
import sys


def show(ind):
    b1=np.reshape(X[ind,:],(28,28))
    plt.imshow(b1,cmap = cm.Greys_r)
    plt.show()


#creating neural net with 2 hiddenlayers
def mlp(input_var):
    l_in=le.layers.InputLayer(shape=(None,1,28,28),input_var=input_var)
    l_in_drop = le.layers.DropoutLayer(l_in, p=0.2)
    l_hid=le.layers.DenseLayer(l_in_drop,num_units=800,nonlinearity=le.nonlinearities.rectify,W=le.init.GlorotUniform())
    l_hid_drop = le.layers.DropoutLayer(l_hid, p=0.5)
    l_hid2=le.layers.DenseLayer(l_hid_drop,num_units=800,nonlinearity=le.nonlinearities.rectify)
    l_hid_drop2 = le.layers.DropoutLayer(l_hid2, p=0.5)
    l_out=le.layers.DenseLayer(l_hid_drop2,num_units=10,nonlinearity=le.nonlinearities.softmax)
    return (l_out)

def cnn(input_var):
    network=le.layers.InputLayer(shape=(None,1,28,28),input_var=input_var)
    network=le.layers.Conv2DLayer(network,num_filters=32,filter_size=(5,5),nonlinearity=le.nonlinearities.rectify,W=le.init.GlorotUniform())
    network=le.layers.MaxPool2DLayer(network,pool_size=(2,2))
    network=le.layers.Conv2DLayer(network,num_filters=32,filter_size=(5,5),nonlinearity=le.nonlinearities.rectify)
    network=le.layers.MaxPool2DLayer(network,pool_size=(2,2))
    network=le.layers.DenseLayer(le.layers.dropout(network,p=0.5),num_units=256,nonlinearity=le.nonlinearities.rectify)
    network=le.layers.DenseLayer(le.layers.dropout(network,p=0.5),num_units=10,nonlinearity=le.nonlinearities.softmax)
    return (network)

#generator function for batch SGD
def iterate_batches(x,y,bsize):
    indices=np.arange(len(x))
    np.random.shuffle(indices)
    for start in range(0,len(x)-bsize+1,bsize):
        slice1=indices[start:start+bsize]
        yield x[slice1],y[slice1]

#loss and update function
if __name__=='__main__':
       
    # import data
    train=pd.read_csv('train.csv').astype('uint8')
    X=train.ix[:,1:].as_matrix()
    X=np.reshape(X,(-1,1,28,28))
    X=X/np.float32(256)
    Y=train.ix[:,0].as_matrix()

    #intializing tensor variables
    input_var = theano.tensor.tensor4('inputs')
    target_var = theano.tensor.ivector('target')

    if (sys.argv[1]=='mlp'):
        network=mlp(input_var)
    else:
        network=cnn(input_var)
    
    prediction=le.layers.get_output(network)
    loss=le.objectives.categorical_crossentropy(prediction,target_var)
    loss=loss.mean()
    acc = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(prediction, axis=1), target_var),dtype=theano.config.floatX)
    params=le.layers.get_all_params(network,trainable=True)
    updates=le.updates.nesterov_momentum(loss,params,learning_rate=0.01,momentum=0.9)

    train_fn=theano.function([input_var,target_var],[loss,acc],updates=updates)
    num_epochs=10

# back-propagation-training weights
    for epoch in range(num_epochs):
        train_er=0
        train_btch=0
        train_acc=0
        for batch in iterate_batches(X,Y,250):
            input,target=batch
            err,ac=train_fn(input,target)
            train_er+=err
            train_acc+=ac
            train_btch+=1
            print ("Epoch {} of {} t".format(epoch+1,num_epochs))
            print("training loss:\t{:.6f}".format(train_er/train_btch))
            print("training accuracy:\t{:.6f}".format(train_acc/train_btch*100))

#saving weights
    params1=le.layers.get_all_param_values(network)
    pickle.dump(params1,open(('params_'+sys.argv[1]+'.p'),'wb'))

    '''  
    to visualize hidden features
    def show_hid(dat):
        b1=np.reshape(dat[3],(5,5))
        plt.imshow(b1,cmap = cm.Greys_r)
        plt.show()

    
    pars=pickle.load(open('params_cnn.p','rb'))
    le.layers.set_all_param_values(network,pars)
    layers=le.layers.get_all_layers(network)
    pred=le.layers.get_output(network,inputs=np.reshape(X[20],(1,1,28,28))).eval()
    print (pred)
    #print(pars[0].shape)
    #show_hid(pars[0])
    #plt.imshow(pred[0,30],cmap = cm.Greys_r)
    #plt.show()
    #show(20)
    '''
