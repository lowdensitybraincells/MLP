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

    buf = f_img.read(8)
    rows, columns = np.frombuffer(buf, dtype=dt, count=2)

    buf = f_img.read(rows*columns*image_count);
    images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

    # converts image to black and white only
    if isBlackAndWhite:
        images = (images>0).astype(np.float32)

    images = images.reshape(image_count,-1)
    images = images/np.max(images)
    labels = np.array([[i==label for i in range(labels.max()+1)] for label in labels], dtype=np.int32) 

    images_train, labels_train = images, labels

# test data reading
    f_label = gzip.open('./data/test/t10k-labels-idx1-ubyte.gz', 'rb')

    dt = dtype(np.int32)
    dt = dt.newbyteorder('>')

    buf = f_label.read(4)
    magic = np.frombuffer(buf, dtype=dt)
    if magic != 2049:
        raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))

    buf = f_label.read(4)
    label_count = np.frombuffer(buf, dtype=dt)[0]

    buf = f_label.read(label_count);
    labels = np.frombuffer(buf, dtype=np.uint8, count=label_count).astype(np.int32)


    # image reading
    f_img = gzip.open('./data/test/t10k-images-idx3-ubyte.gz','rb')

    buf = f_img.read(4)
    magic = np.frombuffer(buf, dtype=dt)
    if magic != 2051:
        raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))

    buf = f_img.read(4)
    image_count = np.frombuffer(buf, dtype=dt)[0]

    buf = f_img.read(8)
    rows, columns = np.frombuffer(buf, dtype=dt, count=2)

    buf = f_img.read(rows*columns*image_count);
    images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

    # converts image to black and white only
    if isBlackAndWhite:
        images = (images>0).astype(np.float32)

    images = images.reshape(image_count,-1)
    images = images/np.max(images)
    labels = np.array([[i==label for i in range(labels.max()+1)] for label in labels], dtype=np.int32) 
    images_test = images
    labels_test = labels
    return images_train, labels_train, images_test, labels_test

