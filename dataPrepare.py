# To do with amazon data
# http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/
import time
import argparse
import numpy as np
from util.utils import constant_sample, get_user_cluster


def prepare_data(user_label_list, train_user_items_dict, test_user_ir_dict, max_dis_dict,
                 nega_user_ratings, num_item, save_root):
    user_label_num = len(set(user_label_list)) - 1
    nega_user_items_rating_dict = {}
    for u_id in nega_user_ratings.keys():
        nega_items_rating_dict = {}
        if nega_user_ratings[u_id] != '':
            for item in nega_user_ratings[u_id].split(','):
                items = item.split(':')
                nega_items_rating_dict[int(items[0])] = float(items[1])
        nega_user_items_rating_dict[u_id] = nega_items_rating_dict.copy()
    # print(nega_user_items_rating_dict[0].keys())
    with open(save_root + '/nega_user_items_rating_dict.txt', 'w') as nega_uir_dict:
        nega_uir_dict.write(str(nega_user_items_rating_dict))

    cluster_items = []  # int, record of items for per cluster
    cluster_users = []  # int, record of users for per cluster
    for i in range(0, user_label_num + 1):  # Initialization
        cluster_users.append(list())
        cluster_items.append(set())
    for user in train_user_items_dict.keys():
        t_label = int(user_label_list[user])
        cluster_users[t_label].append(user)
        for item in train_user_items_dict[user]:
            cluster_items[t_label].add(item)
    with open(save_root + '/cluster_users.txt', 'w') as c_us:
        c_us.write(str({'cluster_users': cluster_users}))

    # Gets the list of classes that appear in the current class but not the farthest from the current class
    supp_nega_cluster_items = {}
    for user_cluster in range(user_label_num + 1):
        train_cluster_items_list = cluster_items[user_cluster].copy()
        max_dis_cluster = max_dis_dict[user_cluster]
        # print(len(cluster_items), len(cluster_items[int_cluster]))
        supp_nega_cluster_items[user_cluster] = cluster_items[max_dis_cluster].copy()
        for train_item in train_cluster_items_list:
            if train_item in cluster_items[max_dis_cluster]:
                supp_nega_cluster_items[user_cluster].remove(train_item)
    with open(save_root + '/supp_nega_cluster_items.txt', 'w') as nega_ci_file:
        nega_ci_file.write(str(supp_nega_cluster_items))

    # Get all test samples in advance for all methods
    test_dict = {}
    test_user_items_rating_dict = {}
    for user_id in test_user_ir_dict.keys():
        user_cluster = int(user_label_list[user_id])
        test_items = set()
        test_items_rating_dict = {}
        for (i_id, rating) in test_user_ir_dict[user_id]:
            test_items.add(i_id)
            test_items_rating_dict[i_id] = float(rating)
        test_user_items_rating_dict[user_id] = test_items_rating_dict.copy()
        test_dict[str(user_id) + '_p'] = list(test_items)

        num_nega = int(len(test_items) / test_percent)
        if num_nega < max_k * 2 - len(test_items):
            num_nega = max_k * 2 - len(test_items)
        # Avoid dead circulation
        if num_nega > num_item - len(train_user_items_dict[user_id]):
            num_nega = num_item - len(train_user_items_dict[user_id])

        if num_nega >= len(supp_nega_cluster_items[user_cluster]):
            negative_list = list(supp_nega_cluster_items[user_cluster].copy())
            while True:
                if len(negative_list) == num_nega:
                    break
                one_negative = np.random.randint(num_item)
                if one_negative not in negative_list:
                    negative_list.append(one_negative)
        else:
            negative_list = constant_sample(supp_nega_cluster_items[user_cluster], num_nega)

        test_dict[str(user_id) + '_n'] = negative_list
    with open(save_root + '/test_dict.txt', 'w') as test_f:
        test_f.write(str(test_dict))
    # print(test_user_items_rating_dict[0].keys())
    with open(save_root + '/test_user_items_rating_dict.txt', 'w') as test_uir_dict:
        test_uir_dict.write(str(test_user_items_rating_dict))


if __name__ == '__main__':
    start_time = time.process_time()  # Starting time
    parser = argparse.ArgumentParser()
    # Data: 'Digital_Music' 'Beauty' 'Clothing_Shoes_and_Jewelry'
    parser.add_argument("-d", dest="data_name", type=str, default='Digital_Music')
    # mf sa sg
    parser.add_argument("-dm", dest="data_method", type=str, default='sg')
    parser.add_argument("-c", dest="cluster_num", type=int, default=5)
    parser.add_argument("-t_p", dest="test_percent", type=float, default=0.1)
    parser.add_argument("-k", dest="max_k", type=int, default=20)
    args = parser.parse_args()
    print(args)

    data_name = args.data_name
    data_method = args.data_method
    save_root = './Data/' + data_name + '/' + data_method + '/'
    test_percent = args.test_percent
    max_k = args.max_k

    item_vectors = np.load(save_root + '/' + data_name + '_embeddings.npy')

    # Classify users according to user vector
    user_vectors = np.load(save_root + '/' + data_name + '_user_embeddings.npy')
    user_lables, max_dis_dict = get_user_cluster(user_vectors, args.cluster_num, save_root, plot=True)

    # Data preparation
    with open(save_root + '/train_user_items_dict.txt', 'r') as f_train_ui_dict:
        train_user_items_dict = eval(f_train_ui_dict.read())
    with open(save_root + '/test_user_ir_dict.txt', 'r') as f_test_user_ir_dict:
        test_user_ir_dict = eval(f_test_user_ir_dict.read())
    with open(save_root + '/nega_user_ratings.txt', 'r', encoding="utf-8") as f:
        nega_user_ratings = eval(f.read())
    with open(save_root + 'data.info', 'r') as info_f:
        d_info = eval(info_f.read())
        item_num = d_info['i_num']
    prepare_data(user_lables, train_user_items_dict, test_user_ir_dict, max_dis_dict, nega_user_ratings, item_num, save_root)

    # End time
    end_time = time.process_time()
    print("Cost time is %f" % (end_time - start_time))
