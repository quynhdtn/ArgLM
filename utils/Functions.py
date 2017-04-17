from theano.tensor.shared_randomstreams import RandomStreams

__author__ = 'quynhdo'
import theano as th
import theano.tensor as T
import numpy as np



import theano.sparse.basic as ST
#### transfer functions

def DotTransferFunction(x, W, b):
        if b is not None:
            return th.dot(x, W) + b
        else:
            return th.dot(x,W)

def spDotTransferFunction(x, W, b):
        if b !=None:
            #return ST.structured_dot(x, W) + b
            return th.dot(x,W) + b
        else:
            return th.dot(x,W)



def NoneTransferFunction(self, x, W, b):
        return x

def MySoftmax(x):
    e_x = T.exp(x - x.max(axis=1, keepdims=True))

    return e_x / e_x.sum(axis=1, keepdims=True)

#### activate functions
SigmoidActivateFunction = T.nnet.sigmoid
SoftmaxActivateFunction =  MySoftmax #T.nnet.softmax
TanhActivateFunction = T.tanh
spSigmoidActivateFunction = ST.structured_sigmoid
spTanhActivateFunction = ST.tanh




def NoneActivateFunction (x):
    return x
def RectifierActivateFunction(x):
			return x*(x>0)

#### cost functions
def NegativeLogLikelihoodCostFunction(o, y, mask=None):
    '''
    Used for Vector output
    :param o: output of the system
    :param y: gold
    :return:
    '''

    if mask is None:
        return -T.mean(T.log(o)[T.arange(y.shape[0]), y])


    else:
        #return -T.mean(T.log(o)[T.arange(y.shape[0]), y] * mask)
        val = T.log(o)[T.arange(y.shape[0]), y]
        val = val * mask
        val = val.sum()
        return -val/mask.sum()

def spNegativeLogLikelihoodCostFunction(o, y):
    '''
    Used for Vector output
    :param o: output of the system
    :param y: gold
    :return:
    '''


    return -T.mean(ST.structured_log(o)[ST.tensor.arange(y.shape[0]), y])

def SquaredErrorCostFunction(o, y, mask=None):
    '''
    used for matrix output
    :param o:
    :param y:
    :return:
    '''
    if mask is None:
        return T.mean((o-y) ** 2)
    else:
        val = (o-y) ** 2
        val = val * mask
        val = val.sum()
        return val/mask.sum()

def SquaredErrorCostFunction2(o, y, mask=None):
    '''
    used for matrix output
    :param o:
    :param y:
    :return:
    '''
    o1 = T.argmax(o, axis=2)
    if mask is None:
        return T.mean((o1-y) ** 2)
    else:
        val = (o1-y) ** 2
        val = val * mask
        val = val.sum()
        return val/mask.sum()


def CrossEntropyCostFunction(o, y, mask=None):
    '''
    used for matrix output
    :param o:
    :param y:
    :return:
    '''
    if mask is None:
        L = - T.sum(y * T.log(o) + (1 - y) * T.log(1 - o), axis=1)

        cost = T.mean(L)
        return cost


def spCrossEntropyCostFunction(o, y):
    '''
    used for matrix output
    :param o:
    :param y:
    :return:
    '''
    L = - ST.sp_sum(y * ST.structured_log(o) + (1 - y) * ST.structured_log(1 - o), axis=1)

    cost = T.mean(L)
    return cost


#### functions to process input for the input layer

def NoneProcessFunction(x, *args): return x



#### init weights

def WeightInit(size_in, size_out, lamda=1, rng=None):
     #rng = np.random.RandomState(89677)
     if rng is None:
        rng = np.random.RandomState(123)
     theano_rng = RandomStreams(rng.randint(2 ** 30))
     return lamda * rng.uniform(
                            low=-1.0,
                            high=1.0,
                            size=(size_in , size_out)
                        )

def getFunction(name):
    if name == "softmax":
        return T.nnet.softmax

    if name == "sigmoid":

        return T.nnet.sigmoid

    if name == "tanh":
        return T.tanh

    if name == "mse":
        return SquaredErrorCostFunction2

    if name == "nl":
        return NegativeLogLikelihoodCostFunction

    if name == "ce":
        return CrossEntropyCostFunction


    if name == "dot_transfer":
        return DotTransferFunction

    if name == "none_transfer":
        return NoneTransferFunction

if __name__ == "__main__":
    x =  np.random.random((3, 2))
    print (x)
    y =   np.random.random(3).astype('int32')
    print(y)
    print (NegativeLogLikelihoodCostFunction(x,y).eval())
    print (NegativeLogLikelihoodCostFunction(x,y, np.ones(3)).eval())