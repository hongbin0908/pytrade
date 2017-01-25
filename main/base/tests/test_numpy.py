import scipy.misc
import matplotlib.pyplot
import numpy as np
def test2_8():
    """
    numpy.ix_
    """
    img_face = scipy.misc.face()
    xmax = img_face.shape[0]
    ymax = img_face.shape[1]
    def shuffle_indices(size):
        arr = np.arange(size)
        np.random.shuffle(arr)
        return arr
    xindices = shuffle_indices(xmax)
    # xindices = np.arange(xmax)
    yindices = shuffle_indices(ymax)
    # yindices = np.arange(ymax)
    idxs = np.ix_(xindices, yindices)
    matplotlib.pyplot.imshow(img_face[idxs])
    matplotlib.pyplot.show()

def test_3_2():
    """
    phi
    """
    phi = (1 + np.sqrt(5))/2
    n=20
    fib = (phi**n - (-1/phi)**n)/np.sqrt(5)
    print(fib.shape)
    assert 0
