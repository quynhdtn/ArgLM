__author__ = 'quynhdo'
import numpy
from theano import config
def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)




def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def ortho_weight(xdim, ydim):
    W = numpy.random.randn(xdim, ydim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def checkSize(lst):
    # lst is list of matrix
    dim = lst[0].shape[1]
    length = 0
    for l in lst:
        if l.shape[0]> length:
            length= l.shape[0]
    return dim,length


def checkSizeAndVob2DList(lst):
    # lst is a list of list
    max_len = 0
    vob = []

    for l in lst:
        for ll in l:
            if not ll in vob:
                vob.append(ll)
        if len(l) > max_len:
            max_len = len(l)

    return len(vob), max_len  # return the size of vob and max len


def Padding(lst, maxlen):
    # lst is a list of matrix 2D
    rs = []
    sw = []
    for l in lst:

        if l.shape[0] <maxlen:
            ll = numpy.concatenate([l,numpy.zeros((maxlen-l.shape[0],l.shape[1]))], axis=0)
            rs.append(ll)
            ww = [1 for  i in range(l.shape[0])]
            for i in range(l.shape[0], maxlen):
                ww.append(0)
            sw.append(ww)
        else:
            rs.append(l)
            xx = [1 for i in range(maxlen)]

            sw.append(xx)
    return numpy.asarray(rs),sw



def prepare_data_seq_classification(seqs, labels, maxlen=None):
        """Create the matrices from the datasets.

        This pad each sequence to the same lenght: the lenght of the
        longuest sequence or maxlen.

        if maxlen is set, we will cut all sequence to this maximum
        lenght.

        This swap the axis!
        """
        # x: a list of sentences
        lengths = [len(s) for s in seqs]

        if maxlen is not None:
            new_seqs = []
            new_labels = []
            new_lengths = []
            for l, s, y in zip(lengths, seqs, labels):
                if l < maxlen:
                    new_seqs.append(s)
                    new_labels.append(y)
                    new_lengths.append(l)
            lengths = new_lengths
            labels = new_labels
            seqs = new_seqs

            if len(lengths) < 1:
                return None, None, None

        n_samples = len(seqs)
        maxlen = numpy.max(lengths)

        x = numpy.zeros((maxlen, n_samples)).astype('int64')
        x_mask = numpy.zeros((maxlen, n_samples)).astype(config.floatX)
        for idx, s in enumerate(seqs):
            x[:lengths[idx], idx] = s
            x_mask[:lengths[idx], idx] = 1.

        return x, x_mask, labels, numpy.ones(len(labels), dtype= 'int32')



def prepare_data_seq(seqs,  maxlen=None):
        """Create the matrices from the datasets.

        This pad each sequence to the same lenght: the lenght of the
        longuest sequence or maxlen.

        if maxlen is set, we will cut all sequence to this maximum
        lenght.

        This swap the axis!
        """
        # x: a list of sentences
        lengths = [len(s) for s in seqs]
        n_samples = len(seqs)
        maxlen = numpy.max(lengths)

        x = numpy.zeros((maxlen, n_samples)).astype('int64')
        x_mask = numpy.zeros((maxlen, n_samples)).astype(config.floatX)
        for idx, s in enumerate(seqs):
            x[:lengths[idx], idx] = s
            x_mask[:lengths[idx], idx] = 1.

        return x.swapaxes(0,1), x_mask.swapaxes(0,1)

