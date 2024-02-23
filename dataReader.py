import numpy as np
import gzip

from numpy.core.multiarray import dtype


def getData(isBlackAndWhite = False):
# label reading
    f_label = gzip.open('./data/train/train-labels-idx1-ubyte.gz', 'rb')

    dt = dtype(np.int32)
    dt = dt.newbyteorder('>')

    buf = f_label.read(4)
    magic = np.frombuffer(buf, dtype=dt)
    if magic != 2049:
        raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))

    buf = f_label.read(4)
    label_count = np.frombuffer(buf, dtype=dt)[0]
    if label_count != 60000:
        raise ValueError('Label count mismatch, expected 60000, got {}'.format(label_count))


    buf = f_label.read(label_count);
    labels = np.frombuffer(buf, dtype=np.uint8, count=label_count).astype(np.int32)


    # image reading
    f_img = gzip.open('./data/train/train-images-idx3-ubyte.gz','rb')

    buf = f_img.read(4)
    magic = np.frombuffer(buf, dtype=dt)
    if magic != 2051:
        raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))

    buf = f_img.read(4)
    image_count = np.frombuffer(buf, dtype=dt)[0]
    if image_count != 60000:
        raise ValueError('Image count mismatch, expected 60000, got {}'.format(label_count))

    buf = f_img.read(8)
    rows, columns = np.frombuffer(buf, dtype=dt, count=2)
    if rows != columns != 60000:
        raise ValueError('Dimension mismatch, expected 28x28 grid, got {}x{}'.format(rows, columns))


    buf = f_img.read(rows*columns*image_count);
    images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

    # converts image to black and white only
    if isBlackAndWhite:
        images = (images>0).astype(np.float32)

    images = images.reshape(image_count,28,28)

    return images, labels
