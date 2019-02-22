import scipy.misc
import numpy as np

def make_one_hot(X):
    # X = np.asarray(X).flatten()
    one_hot = np.zeros((len(X), 10))
    for i in range(len(X)):
        one_hot[i, X[i]] = 1.
    return one_hot

def get_batches(batch_size, data_x, data_y):
    """ Return batch_size of the data 
    vector at a time
    """
    current_index = 0
    while current_index + batch_size <= data_x.shape[0]:
        data_batch_x = data_x[current_index:current_index + batch_size]
        data_batch_y = data_y[current_index:current_index + batch_size]
        current_index += batch_size

        yield data_batch_x, data_batch_y

def save_visualization(X, nh_nw, save_path='./images/sample.jpg'):
    h,w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh_nw[0], w * nh_nw[1], 3))

    for n,x in enumerate(X):
        j = n // nh_nw[1]
        i = n % nh_nw[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = x

    scipy.misc.imsave(save_path, img)

def downsize_real(batch_images, layer_count):
    if layer_count == 2:
        return batch_images
    current_size = int(np.sqrt(batch_images.shape[1]))
    next_size = int(current_size / 2)
    new_arr = np.ndarray((batch_size, next_size**2))
    for i, image in enumerate(batch_images):
        image = image.reshape(current_size, current_size)
        re = image[:next_size*2, :next_size*2].reshape(next_size, 2, next_size, 2).max(axis=(1, 3))
        re = re.reshape(next_size**2)
        new_arr[i] = re
    return downsize_real(new_arr, layer_count +1)

def scale_down_sample(image_tensor, layer_count):
    re = tf.reshape(image_tensor, (batch_size, layer_sizes[layer_count], layer_sizes[layer_count], 1))
    pool = tf.layers.average_pooling2d(re, [2,2], 2)
    back = tf.reshape(pool, (batch_size, layer_sizes[layer_count-1]**2))
    return back

def scale_up_sample(image_tensor, layer_count):
    re = tf.reshape(image_tensor, (batch_size, layer_sizes[layer_count-1],layer_sizes[layer_count-1], 1))
    near_neigh = tf.image.resize_nearest_neighbor(re, size=(layer_sizes[layer_count], layer_sizes[layer_count]))
    back = tf.reshape(near_neigh, [batch_size, layer_sizes[layer_count]**2])
    return back