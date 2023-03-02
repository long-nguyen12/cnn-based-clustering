import numpy as np

def extract_vgg16_features(x):
    from keras.utils import img_to_array, array_to_img
    from keras.applications.vgg16 import preprocess_input, VGG16
    from keras.models import Model

    # im_h = x.shape[1]
    im_h = 224
    model = VGG16(include_top=True, weights='imagenet', input_shape=(im_h, im_h, 3))
    # if flatten:
    #     add_layer = Flatten()
    # else:
    #     add_layer = GlobalMaxPool2D()
    # feature_model = Model(model.input, add_layer(model.output))
    feature_model = Model(model.input, model.get_layer('fc1').output)
    print('extracting features...')
    x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((im_h,im_h))) for im in x])
    x = preprocess_input(x)  # data - 127. #data/255.#
    features = feature_model.predict(x)
    print('Features shape = ', features.shape)

    return features


def make_reuters_data(data_dir):
    np.random.seed(1234)
    from sklearn.feature_extraction.text import CountVectorizer
    from os.path import join
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        # did_to_cat = {k: did_to_cat[k] for k in list(did_to_cat.keys()) if len(did_to_cat[k]) > 1}
        for did in list(did_to_cat.keys()):
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open(join(data_dir, dat)) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did in did_to_cat:
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    print((len(data), 'and', len(did_to_cat)))
    assert len(data) == len(did_to_cat)

    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)

    from sklearn.feature_extraction.text import TfidfTransformer
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
    x = x[:10000].astype(np.float32)
    print(x.dtype, x.size)
    y = y[:10000]
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
    print('todense succeed')

    p = np.random.permutation(x.shape[0])
    x = x[p]
    y = y[p]
    print('permutation finished')

    assert x.shape[0] == y.shape[0]
    x = x.reshape((x.shape[0], -1))
    np.save(join(data_dir, 'reutersidf10k.npy'), {'data': x, 'label': y})


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    print('MNIST samples', x.shape)
    return x, y


def load_fashion_mnist():
    from keras.datasets import fashion_mnist  # this requires keras>=2.0.9
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    print('Fashion MNIST samples', x.shape)
    return x, y


def load_pendigits(data_path='./data/pendigits'):
    import os
    if not os.path.exists(data_path + '/pendigits.tra'):
        os.system('wget http://mlearn.ics.uci.edu/databases/pendigits/pendigits.tra -P %s' % data_path)
        os.system('wget http://mlearn.ics.uci.edu/databases/pendigits/pendigits.tes -P %s' % data_path)
        os.system('wget http://mlearn.ics.uci.edu/databases/pendigits/pendigits.names -P %s' % data_path)

    # load training data
    with open(data_path + '/pendigits.tra') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_train, labels_train = data[:, :-1], data[:, -1]
    print('data_train shape=', data_train.shape)

    # load testing data
    with open(data_path + '/pendigits.tes') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_test, labels_test = data[:, :-1], data[:, -1]
    print('data_test shape=', data_test.shape)

    x = np.concatenate((data_train, data_test)).astype('float32')
    y = np.concatenate((labels_train, labels_test))
    x /= 100.
    print('pendigits samples:', x.shape)
    return x, y


def load_usps(data_path='./data/usps'):
    import os
    if not os.path.exists(data_path+'/usps_train.jf'):
        if not os.path.exists(data_path+'/usps_train.jf.gz'):
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_train.jf.gz -P %s' % data_path)
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_test.jf.gz -P %s' % data_path)
        os.system('gunzip %s/usps_train.jf.gz' % data_path)
        os.system('gunzip %s/usps_test.jf.gz' % data_path)

    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64') / 2.
    y = np.concatenate((labels_train, labels_test))
    print('USPS samples', x.shape)
    return x, y


def load_reuters(data_path='./data/reuters'):
    import os
    if not os.path.exists(os.path.join(data_path, 'reutersidf10k.npy')):
        print('making reuters idf features')
        make_reuters_data(data_path)
        print(('reutersidf saved to ' + data_path))
    data = np.load(os.path.join(data_path, 'reutersidf10k.npy'), allow_pickle=True).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], -1)).astype('float64')
    y = y.reshape((y.size,))
    print(('REUTERSIDF10K samples', x.shape))
    return x, y


def load_retures_keras():
    from keras.preprocessing.text import Tokenizer
    from keras.datasets import reuters
    max_words = 1000

    print('Loading data...')
    (x, y), (_, _) = reuters.load_data(num_words=max_words, test_split=0.)
    print(len(x), 'train sequences')

    num_classes = np.max(y) + 1
    print(num_classes, 'classes')

    print('Vectorizing sequence data...')
    tokenizer = Tokenizer(num_words=max_words)
    x = tokenizer.sequences_to_matrix(x, mode='binary')
    print('x_train shape:', x.shape)

    return x.astype(float), y


def load_imdb():
    from keras.preprocessing.text import Tokenizer
    from keras.datasets import imdb
    max_words = 1000

    print('Loading data...')
    (x1, y1), (x2, y2) = imdb.load_data(num_words=max_words)
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    print(len(x), 'train sequences')

    num_classes = np.max(y) + 1
    print(num_classes, 'classes')

    print('Vectorizing sequence data...')
    tokenizer = Tokenizer(num_words=max_words)
    x = tokenizer.sequences_to_matrix(x, mode='binary')
    print('x_train shape:', x.shape)

    return x.astype(float), y


def load_newsgroups():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.datasets import fetch_20newsgroups
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    vectorizer = TfidfVectorizer(max_features=2000, dtype=np.float64, sublinear_tf=True)
    x_sparse = vectorizer.fit_transform(newsgroups.data)
    x = np.asarray(x_sparse.todense())
    y = newsgroups.target
    print('News group data shape ', x.shape)
    print("News group number of clusters: ", np.unique(y).size)
    return x, y


def load_cifar10(data_path='./data/cifar10'):
    
    # if features are ready, return them
    import os.path
    if os.path.exists(data_path + '/cifar10_features.npy') and os.path.exists(data_path + '/cifar10_labels.npy'):
        return np.load(data_path + '/cifar10_features.npy'), np.load(data_path + '/cifar10_labels.npy')
    
    from keras.datasets import cifar10
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y)).reshape((60000,))

    x = x[:10000]
    y = y[:10000]

    # if features are ready, return them
    # import os.path
    # if os.path.exists(data_path + '/cifar10_features.npy'):
    #     return np.load(data_path + '/cifar10_features.npy'), y

    # extract features
    features = np.zeros((10000, 4096))
    for i in range(1):
        idx = range(i*10000, (i+1)*10000)
        print("The %dth 10000 samples" % i)
        features[idx] = extract_vgg16_features(x[idx])

    # scale to [0,1]
    from sklearn.preprocessing import MinMaxScaler
    features = MinMaxScaler().fit_transform(features)

    # save features
    np.save(data_path + '/cifar10_features.npy', features)
    print('+++  features saved to ' + data_path + '/cifar10_features.npy')

    np.save(data_path + '/cifar10_labels.npy', y)
    print('+++  labels saved to ' + data_path + '/cifar10_labels.npy')

    return features, y


def generate_stl(data_path='./data/stl'):
    import os, sys, tarfile, errno
    import numpy as np
    import matplotlib.pyplot as plt

    import urllib.request
    from imageio import imsave
    from tqdm import tqdm
    import random
    import shutil

    h = 96
    w = 96
    c = 3

    size = h * w * c

    dataset_dir = './data/stl/images'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    train_data_path = './data/stl/train_X.bin'
    train_label_path = './data/stl/train_y.bin'

    test_data_path = './data/stl/test_X.bin'
    test_label_path = './data/stl/test_y.bin'

    unlabeled_data_path = './data/stl/unlabeled_X.bin'

    def read_labels(path_to_labels):
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
            return labels
    
    def read_all_images(path_to_data):
        with open(path_to_data, 'rb') as f:
            everything = np.fromfile(f, dtype=np.uint8)

            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 3, 2, 1))
            return images
    
    def save_image(image, name):
        imsave("%s.png" % name, image, format="png")
    
    def save_images(images, labels, types):
        i = 0
        for image in tqdm(images, position=0):
            label = labels[i] 
            directory = dataset_dir + '/' + types + '/' + str(label) + '/'
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    pass
            filename = directory + str(i)
            #print(filename)
            save_image(image, filename)
            i = i+1

    def save_unlabelled_images(images):
        i = 0
        for image in tqdm(images, position=0):
            directory = dataset_dir + '/' + 'unlabelled' + '/'
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    pass
            filename = directory + str(i)
            save_image(image, filename)
            i = i+1 
    
    def create_val_dataset():
        train_image_path = dataset_dir + "train"
        folders = os.listdir(train_image_path)

        for folder in tqdm(folders, position=0):
            temp_dir = dataset_dir +"/train/" + folder
            temp_image_list = os.listdir(temp_dir)

        for i in range(50):
            val_dir = dataset_dir + "/val/" + folder
            try:
                os.makedirs(val_dir, exist_ok=True)
            except OSError as exc:

                if exc.errno == errno.EEXIST:
                    pass
            image_name = random.choice(temp_image_list)
            temp_image_list.remove(image_name)
            old_name = temp_dir + '/' + image_name
            new_name = val_dir + '/' + image_name
            os.replace(old_name, new_name)
    
    train_labels = read_labels(train_label_path)
    train_images = read_all_images(train_data_path)

    test_labels = read_labels(test_label_path)
    test_images = read_all_images(test_data_path)

    unlabelled_images = read_all_images(unlabeled_data_path)

    save_images(train_images, train_labels, "train")
    save_images(test_images, test_labels, "test")
    save_unlabelled_images(unlabelled_images)


def load_stl():
    from tensorflow import keras
    import numpy as np
    from keras.preprocessing.image import ImageDataGenerator

    # testAug = ImageDataGenerator(rescale = 1./255)
    # train_ds = testAug.flow_from_directory(
	# 	'./data/stl/images/train/',
	# 	class_mode='categorical',
	# 	target_size=(96, 96),
	# 	color_mode="rgb",
	# 	shuffle=False,
	# 	batch_size=32)

    train_ds = keras.utils.image_dataset_from_directory(
        directory='./data/stl/images/train/',
        labels='inferred',
        batch_size=16,
        image_size=(96, 96),
        shuffle=True,
        seed=None,
        validation_split=None,
    ) 

    x = np.concatenate([np.divide(x, 255.0) for x,y in train_ds])
    y = np.concatenate([y for x, y in train_ds], axis=0)
    print(np.unique(y))

    return x, y

def load_data(dataset_name):
    if dataset_name == 'mnist':
        return load_mnist()
    elif dataset_name == 'fmnist':
        return load_fashion_mnist()
    elif dataset_name == 'usps':
        return load_usps()
    elif dataset_name == 'pendigits':
        return load_pendigits()
    elif dataset_name == 'reuters10k' or dataset_name == 'reuters':
        return load_reuters()
    elif dataset_name == 'stl':
        return load_stl()
    elif dataset_name == 'cifar10':
        return load_cifar10()
    else:
        print('Not defined for loading', dataset_name)
        exit(0)

if __name__ == '__main__':
    load_stl()