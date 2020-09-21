import torch
import dgl.function as fn
import torch.nn as nn
import random
import numpy as np
from sklearn.cluster import KMeans


def calculate_distance(vector1, vector2):
    # theta = 0.0000001
    # dist = np.sum(vector1*vector2)/(np.linalg.norm(vector1) * np.linalg.norm(vector2) + theta)  # cosin
    dist = np.sum(np.square(vector1 - vector2))  # Euclidean distance
    return dist


def clean_data(texts: str):
    # lowdown cased
    texts = texts.lower()
    texts = texts.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ').replace('*', ' ') \
        .replace('-', '').replace('......', ' ').replace('...', ' ').replace('?', ' ').replace(':', ' ') \
        .replace(';', ' ').replace(',', ' ').replace('.', ' ').replace('!', ' ').replace("/", ' ') \
        .replace('" ', ' ').replace("' ", ' ').replace(' "', ' ').replace(" '", ' ').replace("=", ' ') \
        .replace('  ', ' ').replace('  ', ' ')
    return texts


# Constant time complexity instead of sample
def constant_sample(o_list, sample_num: int):
    sample_list = []
    in_list = list(o_list).copy()
    for i in range(sample_num):
        r_n = random.randint(0, len(in_list) - 1)
        # print(r_n, len(in_list))
        sample_list.append(in_list[r_n])
        in_list.pop(r_n)
    return sample_list


class HeteroGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etypes
        })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HeteroGCNNet(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size,
                 use_dr_pre, pre_v_dict):
        super(HeteroGCNNet, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        if use_dr_pre:
            embed_dict = {}
            for ntype in G.ntypes:
                if ntype == 'description' or ntype == 'review':
                    pre_vec = torch.tensor(pre_v_dict[ntype], dtype=torch.float)
                    assert pre_vec.shape[0] == G.number_of_nodes(ntype), 'The shape of pre_vec is wrong!'
                    embed = nn.Parameter(pre_vec)
                    embed_dict[ntype] = embed
                else:
                    embed = nn.Parameter(torch.zeros(G.number_of_nodes(ntype), in_size))
                    nn.init.xavier_uniform_(embed)
                    embed_dict[ntype] = embed
        else:
            embed_dict = {}
            for ntype in G.ntypes:
                embed = nn.Parameter(torch.zeros(G.number_of_nodes(ntype), in_size))
                nn.init.xavier_uniform_(embed)
                embed_dict[ntype] = embed
        self.embed = nn.ParameterDict(embed_dict)
        # create layers
        self.layer1 = HeteroGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroGCNLayer(hidden_size, out_size, G.etypes)
        self.h_dict = None

    def forward(self, G):
        self.h_dict = self.layer1(G, self.embed)
        # print(self.h_dict)
        self.h_dict = self.layer2(G, self.h_dict)
        # print(self.h_dict)

    def lookup_emb_list(self, name: str, index_list: list = None, need_all=False):
        if need_all:
            return self.h_dict[name]
        else:
            emb = torch.zeros(len(index_list), self.h_dict[name].shape[1])
            i = 0
            for index in index_list:
                emb[i] = self.h_dict[name][index]
                i += 1
            # print(emb, emb.shape)
        return emb

    def lookup_emb(self, name: str, emb_index: int):
        return self.h_dict[name][emb_index]


def get_user_cluster(i_user_data, cluster_num: int, save_root, plot=False):
    estimator = KMeans(n_clusters=cluster_num, max_iter=500)  # Construct cluster
    estimator.fit(i_user_data)
    label_predict = estimator.labels_  # Get cluster labels
    class_center = estimator.cluster_centers_
    # Save K-means results
    # print(len(label_predict))
    with open(save_root + "/user_label_predict.txt", 'w+') as l_file:
        str_label = str(list(label_predict)).replace('[', '').replace(']', '').replace(' ', '')
        l_file.write(str_label)
    # Calculate the distance between the classification center
    # select the one furthest from the cluster center, and save it
    # Calculate the distance between class pairs
    dis_pair_list = list()
    i = 0
    while i < class_center.shape[0] - 1:
        j = i + 1
        while j < class_center.shape[0]:
            dis_pair_list.append([calculate_distance(class_center[i], class_center[j]), [i, j]])
            j += 1
        i += 1
    # print(dis_pair_list)
    # Get the maximum distance for each center
    large_dis_center = list()
    i = 0
    while i < class_center.shape[0]:
        large_pair = None
        max_dis = -1
        for one_pair in dis_pair_list:
            if i in one_pair[1]:
                if one_pair[0] > max_dis:
                    max_dis = one_pair[0]
                    large_pair = one_pair[1]
        large_dis_center.append(large_pair)
        i += 1
    # print(np.array(large_dis_center))
    max_dis_dict = {}
    i = 0
    for t_cluster in large_dis_center:
        if i == t_cluster[0]:
            max_dis_dict[i] = t_cluster[1]
        else:
            max_dis_dict[i] = t_cluster[0]
        i += 1
    # print(class_center)
    np.save(save_root + '/class_center.npy', class_center)
    # print(np.array(large_dis_center))
    with open(save_root + '/max_dis_pair.txt', 'w', encoding="utf-8") as f:
        f.write(str(max_dis_dict))
    if plot:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        t_pca = PCA(n_components=2)
        low_dim_embs = t_pca.fit_transform(i_user_data)
        # Draw K-means results
        for i in range(0, cluster_num):
            x = low_dim_embs[label_predict == i]
            plt.scatter(x[:, 0], x[:, 1], label='cluster' + str(i))
        plt.legend(loc='best')
        plt.show()
        plt.close()
    return label_predict, max_dis_dict


def mf_with_bias(data_shape, emb_size, rating_list, lr=1e-2, l2_factor=1e-2, max_step=1000, train_rate=0.95, max_stop_count=30):
    import tensorflow as tf
    rating = np.array(rating_list)
    user_num = data_shape[0]
    item_num = data_shape[1]
    boundry_user_id = int(user_num * 0.8)
    print('training pmf...')

    data = np.array(list(filter(lambda x: x[0] < boundry_user_id, rating)))
    np.random.shuffle(data)

    t = int(len(data)*train_rate)
    dtrain = data[:t]
    dtest = data[t:]

    user_embeddings = tf.Variable(tf.truncated_normal([user_num, emb_size], mean=0, stddev=0.01))
    item_embeddings = tf.Variable(tf.truncated_normal([item_num, emb_size], mean=0, stddev=0.01))
    item_bias = tf.Variable(tf.zeros([item_num, 1], tf.float32))

    user_ids = tf.placeholder(tf.int32, shape=[None])
    item_ids = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    user_embs = tf.nn.embedding_lookup(user_embeddings, user_ids)
    item_embs = tf.nn.embedding_lookup(item_embeddings, item_ids)
    ibias_embs = tf.nn.embedding_lookup(item_bias, item_ids)
    dot_e = user_embs * item_embs

    ys_pre = tf.reduce_sum(dot_e, 1)+tf.squeeze(ibias_embs)

    target_loss = tf.reduce_mean(0.5 * tf.square(ys - ys_pre))
    loss = target_loss + l2_factor * (tf.reduce_mean(tf.square(user_embs) + tf.square(item_embs)))

    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - ys_pre)))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.random.shuffle(dtrain)
        rmse_train, loss_v, target_loss_v = sess.run([rmse, loss, target_loss], feed_dict={user_ids: dtrain[:, 0], item_ids: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        rmse_test = sess.run(rmse, feed_dict={user_ids: dtest[:, 0], item_ids: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print('----round%2d: rmse_train: %f, rmse_test: %f, loss: %f, target_loss: %f' % (0, rmse_train, rmse_test, loss_v, target_loss_v))
        pre_rmse_test = 100.0
        stop_count = 0
        stop_count_flag = False
        for i in range(max_step):
            feed_dict = {user_ids: dtrain[:, 0],
                         item_ids: dtrain[:, 1],
                         ys: np.float32(dtrain[:, 2])}
            sess.run(train_step, feed_dict)
            np.random.shuffle(dtrain)
            rmse_train, loss_v, target_loss_v = sess.run([rmse, loss, target_loss], feed_dict={user_ids: dtrain[:, 0], item_ids: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
            rmse_test = sess.run(rmse, feed_dict={user_ids: dtest[:, 0], item_ids: dtest[:, 1], ys: np.float32(dtest[:, 2])})
            print('----round%2d: rmse_train: %f, rmse_test: %f, loss: %f, target_loss: %f' % (i + 1, rmse_train, rmse_test, loss_v, target_loss_v))
            if rmse_test>pre_rmse_test:
                stop_count += 1
                if stop_count==max_stop_count:
                    stop_count_flag = True
                    break
            pre_rmse_test = rmse_test

        return sess.run(item_embeddings)


def get_user_vector(user_ir_dict: dict, u_num: int, i_movie_vectors, embedding_size):
    user_embeddings = np.zeros((u_num, embedding_size))
    user_emb = np.zeros(embedding_size)
    for user in user_ir_dict.keys():
        for (item, rating) in user_ir_dict[user]:
            user_emb = user_emb + i_movie_vectors[item]
        user_embeddings[user] = user_emb / len(user_ir_dict[user])  # normalise
        user_emb = np.zeros(embedding_size)
    print('All user embeddings done.')
    return user_embeddings
