
from math import gamma
from time import time
import numpy as np
from keras.models import Model
from keras.optimizers import SGD
# from keras.utils.vis_utils import plot_model

from keras import callbacks
from sklearn.cluster import KMeans

from DEC import ClusteringLayer, autoencoder
import metrics
from keras.initializers import VarianceScaling

path_save = ""
class IDEC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=256,
                 init='glorot_uniform'):

        super(IDEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.autoencoder, self.encoder = autoencoder(self.dims, init=init)

        # prepare IDEC model
        clustering_layer = ClusteringLayer(
            self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=[
                           clustering_layer, self.autoencoder.output])

    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256, save_dir='results/temp'):
        print('...Pretraining..., batch_size = ', batch_size)
        self.autoencoder.summary()
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        # logging file
        import csv
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_idec_log.csv')
        cb = [csv_logger]
        if y is not None:
            class PrintACC(callbacks.Callback):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    super(PrintACC, self).__init__()

                def on_epoch_end(self, epoch, logs=None):
                    if int(epochs/10) != 0 and epoch % int(epochs/10) != 0:
                        return
                    feature_model = Model(self.model.input,
                                          self.model.get_layer(
                                              'encoder_%d' % (int(len(self.model.layers) / 2) - 1)).output)
                    features = feature_model.predict(self.x)
                    km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20)
                    y_pred = km.fit_predict(features)
                    print(self.y, np.unique(self.y), len(np.unique(self.y)),
                          'encoder_%d' % (int(len(self.model.layers) / 2) - 1))
                    print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                          % (metrics.acc(self.y, y_pred), metrics.nmi(self.y, y_pred)))

            cb.append(PrintACC(x, y))

        # begin pretraining
        t0 = time()
        self.autoencoder.fit(x, x, batch_size=batch_size,
                             epochs=epochs, callbacks=cb)
        print('Pretraining time: %ds' % round(time() - t0))
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights_path):  # load weights of IDEC model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    # predict cluster labels using the output of clustering layer
    def predict_clusters(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    # target distribution P which enhances the discrimination of soft label Q
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss={'clustering': 'kld', 'decoder_0': 'mse'}, gamma=0.1):

        self.model.compile(loss={'clustering': 'kld', 'decoder_0': 'mse'},
                           loss_weights=[0.8,  0.2], optimizer=optimizer)
        #https://stackoverflow.com/questions/49583805/using-different-loss-functions-for-different-outputs-simultaneously-keras

    def clustering(self, x, y=None,
                   update_interval=30,
                   maxiter=5000,
                   batch_size=64,
                   save_dir='./results/idec'):

        print('Update interval', update_interval)

        # Step 1: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])       

        # logging file
        import csv
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/idec_log.csv', 'w')
        logwriter = csv.DictWriter(
            logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        loss = [0, 0, 0]
        index = 0
        index_array = np.arange(x.shape[0])
        print('++++ index_array:',index_array)  
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q,_ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss, 'bs=',batch_size, 'max_iter=', maxiter, 'update_interval:', update_interval)

            # train on batch, 
            # if index == 0:
            #     np.random.shuffle(index_array)     
            #     print(index_array)          
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            loss = self.model.train_on_batch(x=x[idx], y=[p[idx], x[idx]])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
           

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/IDEC_model_final.h5')
        self.model.save_weights(save_dir + '/IDEC_model_final.h5')

        return y_pred


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='mnist',
                        choices=['mnist', 'fmnist', 'usps', 'reuters10k', 'stl', 'cifar10'])
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=5000, type=int)
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--pretrain_epochs', default=None, type=int)
    parser.add_argument('--update_interval', default=140, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None,
                        help='This argument must be given')
    parser.add_argument('--save_dir', default='results/idec')
    args = parser.parse_args()
    print(args)
    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    from datasets import load_data
    x, y = load_data(args.dataset)
    n_clusters = len(np.unique(y))

    init = 'glorot_uniform'
    pretrain_optimizer = 'adam'
    bs = 64
    update_interval = 30
    pretrain_epochs = 300
    num_clusters = 10

    # setting parameters
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        update_interval = 140
        pretrain_epochs = 300
        init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
        pretrain_optimizer = SGD(lr=1, momentum=0.9)
        bs = 256
    elif args.dataset == 'reuters10k':
        update_interval = 30
        pretrain_epochs = 50
        init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
        pretrain_optimizer = SGD(lr=1, momentum=0.9)
        bs = 64
    elif args.dataset == 'usps':
        update_interval = 30
        pretrain_epochs = 50
        
    elif args.dataset == 'stl':
        update_interval = 30
        pretrain_epochs = 30
        bs = 64
    elif args.dataset == 'cifar10':
        update_interval = 30
        pretrain_epochs = 20
    
    path_save = "results/" + args.dataset + "/idec"
    print('+ path_save = ', path_save)
    
    # prepare the IDEC model
    idec = IDEC(dims=[x.shape[-1], 200, 200, 400, 20], n_clusters=num_clusters, init=init)

    if args.ae_weights is None:
        idec.autoencoder.load_weights(path_save + '/ae_weights.h5')
        #idec.pretrain(x=x, y=y, optimizer=pretrain_optimizer, epochs=pretrain_epochs, batch_size=args.batch_size, save_dir=path_save)
    else:
        idec.autoencoder.load_weights(path_save + '/ae_weights.h5')

    idec.model.summary()    
    t0 = time()
    idec.compile(optimizer=SGD(0.01, 0.9), loss={'clustering': 'kld', 'decoder_0': 'mse'}, gamma=0.1)
    # begin clustering, time not include pretraining part.
    
    y_pred = idec.clustering(x, y=y, maxiter=args.maxiter, batch_size=bs, update_interval=update_interval, save_dir=path_save)
    print('acc:', metrics.acc(y, y_pred))
    print('clustering time: ', (time() - t0))
