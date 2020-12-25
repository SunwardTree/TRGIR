# To do with amazon data
# http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/

import os
import math
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from util.utils import clean_data
from sentence_transformers import models, SentenceTransformer


def restore_data(d_name: str, save_root, positive_ratings: int, get_true_scale=False):
    f_name = data_name + '_5.json'
    old_user2new = {}
    old_item2new = {}
    new_data = []
    train_data = []
    ne_datas = []
    user_ir_dict = {}
    nega_user_items = {}
    nega_user_ratings = {}
    ui_pair, iu_pair = [], []
    ui_r_dict = {}
    reviews_dict = {}

    current_u_index = 0
    current_i_index = 0
    current_r_index = 0
    rating_counts = 0
    user_list = []
    item_list = []
    valid_rating = 0
    with open('./Data/' + d_name + '/' + f_name, 'r', encoding="utf-8") as f:
        while True:
            temp_str = f.readline()
            if temp_str:
                temp_dict = eval(temp_str)
                o_user = temp_dict['reviewerID']
                o_item = temp_dict['asin']
                if get_true_scale:
                    if o_user not in user_list:
                        user_list.append(o_user)
                    if o_item not in item_list:
                        item_list.append(o_item)
                rating = float(temp_dict['overall'])
                rating_counts += 1
                if rating >= positive_ratings:
                    valid_rating += 1
                    try:
                        n_user = old_user2new[o_user]
                        # print(n_user)
                    except KeyError:
                        n_user = current_u_index
                        old_user2new[o_user] = current_u_index
                        current_u_index += 1
                    try:
                        n_item = old_item2new[o_item]
                        # print(n_item)
                    except KeyError:
                        n_item = current_i_index
                        old_item2new[o_item] = current_i_index
                        current_i_index += 1

                    times = int(temp_dict['unixReviewTime'])
                    try:
                        reviews = clean_data(str(temp_dict['reviewText']))
                        reviews += clean_data(str(temp_dict['summary']))
                    except KeyError:
                        try:
                            reviews = clean_data(str(temp_dict['summary']))
                        except KeyError:
                            continue
                    new_data.append((n_user, n_item, rating, times))
                    train_data.append((n_user, n_item, rating))
                    reviews_dict[current_r_index] = reviews
                    ui_r_dict[(n_user, n_item)] = current_r_index
                    current_r_index += 1
                else:
                    ne_datas.append((o_user, o_item, rating))
            else:
                break
    new_data = sorted(new_data, key=lambda x: (x[0], x[3]))  # Sort data by user_id and time
    with open(save_root + '/n_rating.txt', 'w', encoding="utf-8") as f:
        t_str = str(new_data)
        t_str = t_str[:len(t_str) - 2].replace('[', '').replace(']', '').replace('(', '').replace(' ', '') \
            .replace('),', '\n')
        f.write(t_str)
    for t_d in new_data:
        ui_pair.append((t_d[0], t_d[1]))
        iu_pair.append((t_d[1], t_d[0]))
        try:
            user_ir_dict[t_d[0]].append((t_d[1], t_d[2]))
        except KeyError:
            user_ir_dict[t_d[0]] = [(t_d[1], t_d[2])]

    for ne_data in ne_datas:
        try:
            new_user_id = old_user2new[ne_data[0]]
        except KeyError:
            continue
        try:
            new_item_id = old_item2new[ne_data[1]]
        except KeyError:
            continue
        try:
            nega_user_items[new_user_id] = nega_user_items[new_user_id] + ',' + str(new_item_id)
            nega_user_ratings[new_user_id] = nega_user_ratings[new_user_id] + ',' + str(new_item_id) + ':' + str(ne_data[2])
        except KeyError:
            nega_user_items[new_user_id] = str(new_item_id)
            nega_user_ratings[new_user_id] = str(new_item_id) + ':' + str(ne_data[2])
    with open(save_root + '/nega_user_items.txt', 'w', encoding="utf-8") as f:
        f.write(str(nega_user_items))
    with open(save_root + '/nega_user_ratings.txt', 'w', encoding="utf-8") as f:
        f.write(str(nega_user_ratings))

    with open(save_root + '/old_user2new.txt', 'w', encoding="utf-8") as f:
        f.write(str(old_user2new))
    with open(save_root + '/old_item2new.txt', 'w', encoding="utf-8") as f:
        f.write(str(old_item2new))
    if get_true_scale:
        print('user_counts:', len(user_list))
        print('item_counts:', len(item_list))
    print('valid_user:', current_u_index)
    print('valid_item:', current_i_index)
    print('rating_counts', rating_counts)
    print('valid_rating:', valid_rating)
    return new_data, train_data, user_ir_dict, ui_pair, iu_pair, ui_r_dict, reviews_dict, \
           old_user2new, old_item2new, current_u_index, current_i_index, current_r_index, nega_user_ratings


def get_descriptions(d_name: str, old_item2new):
    descriptions_dict = {}
    des_index = 0
    di_pair = []
    id_pair = []
    des_path = './Data/' + d_name + '/meta_' + d_name + '.json'  # descriptions
    with open(des_path, 'r', encoding="utf-8") as f:
        while True:
            temp_str = f.readline()
            if temp_str:
                temp_dict = eval(temp_str)
                t_asin = temp_dict['asin']
                try:
                    n_item = old_item2new[t_asin]
                    t_descriptions = clean_data(str(temp_dict['description']))
                    t_descriptions += clean_data(str(temp_dict['categories']))
                except KeyError:
                    try:
                        n_item = old_item2new[t_asin]
                        t_descriptions = clean_data(str(temp_dict['categories']))
                    except KeyError:
                        continue
                descriptions_dict[n_item] = t_descriptions
                di_pair.append((des_index, n_item))
                id_pair.append((n_item, des_index))
                des_index += 1
            else:
                break
    print('Get ', des_index, ' ' + d_name + ' descriptions.')
    return des_index, di_pair, id_pair, descriptions_dict


def get_train_test(user_ir_dict, test_percent: float, more_than_one: bool):
    train_user_ir_dict = {}
    test_user_ir_dict = {}
    train_user_items_dict = {}
    train_user_items_rating_dict = {}

    for u_id in user_ir_dict.keys():
        num_items = len(user_ir_dict[u_id])
        t_s_num = int(np.round(num_items * test_percent))
        select_num = num_items - t_s_num
        if more_than_one and num_items > 1:
            if t_s_num == 0:
                t_s_num = 1
                select_num = num_items - t_s_num
        if select_num < num_items:
            test_user_ir_dict[u_id] = user_ir_dict[u_id][num_items - t_s_num:]
        train_user_ir_dict[u_id] = user_ir_dict[u_id][:num_items - t_s_num]
        train_items_list = []
        train_items_rating_dict = {}
        for (i_id, rating) in train_user_ir_dict[u_id]:
            train_items_list.append(i_id)
            train_items_rating_dict[i_id] = rating
        train_user_items_dict[u_id] = train_items_list.copy()
        train_user_items_rating_dict[u_id] = train_items_rating_dict.copy()
    # print(train_user_items_rating_dict)
    # print(train_user_items_dict)
    with open(save_root + '/train_user_items_dict.txt', 'w') as f_train_ui_dict:
        f_train_ui_dict.write(str(train_user_items_dict))
    with open(save_root + '/train_user_items_rating_dict.txt', 'w') as f_train_uir_dict:
        f_train_uir_dict.write(str(train_user_items_rating_dict))
    with open(save_root + '/test_user_ir_dict.txt', 'w') as f_test_user_ir_dict:
        f_test_user_ir_dict.write(str(test_user_ir_dict))
    print('Get train and test split by time-line.')
    return train_user_items_dict, test_user_ir_dict


def get_stop_word(stop_word_path: str):
    with open(stop_word_path) as stop_word_file:
        i_stop_words_list = (stop_word_file.read()).split()
    return i_stop_words_list


def get_glove_dict(glove_dict_path: str):
    with open(glove_dict_path, 'r', encoding="utf-8") as glove_file:
        in_glove_dict = {}
        t_list = []
        for line in glove_file.readlines():
            t_list = line.split()
            if len(t_list) > 1:
                tt_list = []
                for number in t_list[1:]:
                    tt_list.append(float(number))
                in_glove_dict[t_list[0]] = np.array(tt_list)
    return in_glove_dict


def get_vector(record_texts_dict: dict, embedding_size, record_num: int, device,
               save_path='', s_text_emb_method='glove'):
    record_embeddings = None
    if s_text_emb_method == 'glove':
        print('----glove vector----')
        stop_word_list = get_stop_word(stop_word_path='./resource/stop_words.txt')
        glove_dict = get_glove_dict(glove_dict_path='./resource/glove/glove.6B.' + str(embedding_size) + 'd.txt')
        record_embeddings = np.zeros((record_num, embedding_size))
        t_count = 0
        # print(item_num)
        for i in tqdm(range(record_num)):
            item_emb = np.zeros(embedding_size)
            try:
                word_str = str(record_texts_dict[i])
                word_list = word_str.split(" ")
                # print(word_list)
                t_div = 1
                for word in word_list:
                    if word not in stop_word_list:
                        try:
                            word_glove_vector = glove_dict[word]
                            item_emb = item_emb + word_glove_vector
                        except KeyError:
                            continue
                        t_div += 1
                    else:
                        continue
                # print(t_div, item_emb, item_emb / t_div)
                record_embeddings[i] = item_emb / t_div  # normalise
                t_count += 1
            except KeyError:
                continue
    elif s_text_emb_method == 'sbert':
        print('----sentence-bert vector----')
        # Sentence-BERT:
        # Sentence Embeddings using Siamese BERT-Networks https://arxiv.org/abs/1908.10084
        # https://github.com/UKPLab/sentence-transformers
        # google/bert_uncased_L-2_H-128_A-2(BERT-Tiny)
        # google/bert_uncased_L-12_H-256_A-4(BERT-Mini)
        # google/bert_uncased_L-4_H-512_A-8(BERT-Small)
        # google/bert_uncased_L-8_H-512_A-8(BERT-Medium)
        # google/bert_uncased_L-12_H-768_A-12(BERT-Base)
        word_embedding_model = models.BERT('google/bert_uncased_L-12_H-256_A-4', max_seq_length=510)
        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        bert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
        one_req_num = 500
        record_list = list(record_texts_dict.values())
        req_times = int(math.ceil(len(record_list) / one_req_num))
        for ii in tqdm(range(req_times)):
            if ii == 0:
                record_embeddings = bert_model.encode(record_list[ii * one_req_num: (ii + 1) * one_req_num])
            elif ii < req_times - 1:
                record_embeddings = np.vstack(
                    (record_embeddings, bert_model.encode(record_list[ii * one_req_num: (ii + 1) * one_req_num])))
            else:
                record_embeddings = np.vstack((record_embeddings, bert_model.encode(record_list[ii * one_req_num:])))
    else:
        print('Do not support', s_text_emb_method, 'text embedding method.')
    if save_path != '':
        np.save(save_path, record_embeddings)
    return record_embeddings


def get_x_pairs(con_matrix, the_id, num_xs, num_pairs, if_pos=True):
    x_pairs = []
    pairs_add = x_pairs.append
    while True:
        xx_id = np.random.randint(num_xs)
        if not if_pos and con_matrix[the_id][xx_id] == 0:
            pairs_add((the_id, xx_id))
        elif if_pos and con_matrix[the_id][xx_id] != 0:
            pairs_add((the_id, xx_id, con_matrix[the_id][xx_id]))
        if len(x_pairs) == num_pairs:
            break
    return x_pairs


if __name__ == '__main__':
    start_time = time.process_time()  # Starting time
    parser = argparse.ArgumentParser()
    # Data: 'Digital_Music' 'Beauty' 'Clothing_Shoes_and_Jewelry'
    parser.add_argument("-d", dest="data_name", type=str, default='Digital_Music')
    # mf sa sg
    parser.add_argument("-dm", dest="data_method", type=str, default='sg')
    # glove sbert
    parser.add_argument("-stm", dest="s_t_emb_method", type=str, default='sbert')
    parser.add_argument("-active", dest="if_non_active", type=bool, default=True)
    parser.add_argument("-t_p", dest="test_percent", type=float, default=0.1)
    parser.add_argument("-r_b", dest="rating_bound", type=float, default=3)
    parser.add_argument("-k", dest="max_k", type=int, default=20)
    parser.add_argument("-e", dest="epochs", type=int, default=100)
    parser.add_argument("-b", dest="batch_size", type=int, default=256)
    parser.add_argument("-emb_size", dest="emb_size", type=int, default=100)
    parser.add_argument("-gout", dest="graph_out_dim", type=int, default=64)
    parser.add_argument("-ghid", dest="graph_hidden_dim", type=int, default=128)
    parser.add_argument("-loop", dest="self_loop", type=int, default=1, help='1 is True.')
    parser.add_argument("-p", dest="dr_pre_train", type=int, default=1, help='1 is True.')
    parser.add_argument("-u_d", dest="use_des", type=int, default=1, help='1 is True.')
    parser.add_argument("-u_r", dest="use_rev", type=int, default=1, help='1 is True.')
    parser.add_argument("-gpu_id", dest="gpu_id", type=int, default=2, help='If use cpu, set -1.')
    args = parser.parse_args()
    print(args)

    data_name = args.data_name
    data_method = args.data_method
    s_t_emb_method = args.s_t_emb_method
    if data_method == 'mf':
        s_t_emb_method = ''
    if_non_active = args.if_non_active
    test_percent = args.test_percent
    rating_bound = args.rating_bound
    epochs = args.epochs
    max_k = args.max_k
    save_root = './Data/' + data_name + '/' + data_method + '/' + s_t_emb_method + '/'
    if not os.path.exists(save_root): os.makedirs(save_root)
    graph_in_dim = args.emb_size
    graph_out_dim = args.graph_out_dim
    graph_hidden_dim = args.graph_hidden_dim
    self_loop = True if args.self_loop == 1 else False
    use_des = True if args.use_des == 1 else False
    use_rev = True if args.use_rev == 1 else False
    dr_pre_train = True if args.dr_pre_train == 1 else False
    if not (use_des or use_rev):
        dr_pre_train = False
    if not dr_pre_train:
        use_des, use_rev = False, False
    gpu_id = args.gpu_id

    new_data, train_data, user_ir_dict, ui_pair, iu_pair, ui_r_dict, old_reviews_dict, \
    old_user2new, old_item2new, user_num, item_num, rev_num, nega_user_ratings = \
        restore_data(data_name, save_root, rating_bound, get_true_scale=False)
    train_user_items_dict, test_user_ir_dict = get_train_test(user_ir_dict, test_percent, more_than_one=True)

    with open(save_root + 'data.info', 'w') as data_f:
        d_info = {'u_num': user_num, 'i_num': item_num}
        data_f.write(str(d_info))

    for tt_uid in test_user_ir_dict.keys():
        for (tt_iid, rating) in test_user_ir_dict[tt_uid]:
            ui_pair.remove((tt_uid, tt_iid))
            iu_pair.remove((tt_iid, tt_uid))
            train_data.remove((tt_uid, tt_iid, rating))
            old_reviews_dict.pop(ui_r_dict[(tt_uid, tt_iid)])
            rev_num -= 1
    # review 重排序
    new_rev_id = 0
    old_rev_id2new = {}
    reviews_dict = {}
    for old_rev_id in old_reviews_dict.keys():
        old_rev_id2new[old_rev_id] = new_rev_id
        reviews_dict[new_rev_id] = old_reviews_dict[old_rev_id]
        new_rev_id += 1
    del old_reviews_dict

    str_dev = "cuda:" + str(gpu_id)
    device = torch.device(str_dev if (torch.cuda.is_available() and gpu_id >= 0) else "cpu")
    print('Device:', device)

    emb_size = args.emb_size
    if data_method == 'mf':
        from util.utils import mf_with_bias, get_user_vector
        item_vectors = mf_with_bias([user_num, item_num], emb_size, train_data)
        user_vectors = get_user_vector(user_ir_dict, user_num, item_vectors, emb_size)
    elif data_method == 'sa':
        from util.utils import get_user_vector
        alpha = 0.5
        _, _, _, des_dict = get_descriptions(data_name, old_item2new)
        # get vectors
        description_vectors = get_vector(des_dict, emb_size, item_num, device, s_text_emb_method=s_t_emb_method)
        review_vectors = get_vector(reviews_dict, emb_size, item_num, device, s_text_emb_method=s_t_emb_method)
        item_vectors = alpha * description_vectors + (1 - alpha) * review_vectors
        user_vectors = get_user_vector(user_ir_dict, user_num, item_vectors, emb_size)
    elif data_method == 'sg':
        import dgl
        from util.utils import HeteroGCNNet

        ur_pair, ru_pair = [], []
        ir_pair, ri_pair = [], []
        for t_uid in train_user_items_dict.keys():
            for t_iid in train_user_items_dict[t_uid]:
                t_tuple = (t_uid, t_iid)
                ur_pair.append((t_uid, old_rev_id2new[ui_r_dict[t_tuple]]))
                ru_pair.append((old_rev_id2new[ui_r_dict[t_tuple]], t_uid))
                ir_pair.append((t_iid, old_rev_id2new[ui_r_dict[t_tuple]]))
                ri_pair.append((old_rev_id2new[ui_r_dict[t_tuple]], t_iid))

        des_num, di_pair, id_pair, descriptions_dict = get_descriptions(data_name, old_item2new)
        uu_pair = list(zip(list(range(user_num)), list(range(user_num))))
        ii_pair = list(zip(list(range(item_num)), list(range(item_num))))
        dd_pair = list(zip(list(range(des_num)), list(range(des_num))))
        rr_pair = list(zip(list(range(rev_num)), list(range(rev_num))))
        # print('ui_pair:', len(ui_pair), ', ur_pair:', len(ur_pair),
        #       ', ri_pair:', len(ri_pair), ', di_pair:', len(di_pair),
        #       ', uu_pair:', len(uu_pair), ', ii_pair:', len(ii_pair),
        #       ', dd_pair:', len(dd_pair), ', rr_pair:', len(rr_pair))
        graph_info = {'ui_pair': ui_pair, 'iu_pair': iu_pair, 'ur_pair': ur_pair, 'ru_pair': ru_pair,
                      'ri_pair': ri_pair, 'ir_pair': ir_pair, 'di_pair': di_pair, 'id_pair': id_pair,
                      'uu_pair': uu_pair, 'ii_pair': ii_pair, 'dd_pair': dd_pair, 'rr_pair': rr_pair}
        with open(save_root + 'graph.info', 'w') as info_f:
            info_f.write(str(graph_info))

        if use_des and use_rev:
            print('graph_in_dim reset as', graph_in_dim)
            des_vectors = get_vector(descriptions_dict, emb_size, des_num, device, s_text_emb_method=s_t_emb_method)
            rev_vectors = get_vector(reviews_dict, emb_size, rev_num, device, s_text_emb_method=s_t_emb_method)
            pre_vec_dict = {'description': des_vectors, 'review': rev_vectors}
            graph_in_dim = pre_vec_dict['description'].shape[1]
        elif use_des:
            print('graph_in_dim reset as', graph_in_dim)
            des_vectors = get_vector(descriptions_dict, emb_size, des_num, device, s_text_emb_method=s_t_emb_method)
            pre_vec_dict = {'description': des_vectors}
            graph_in_dim = pre_vec_dict['description'].shape[1]
        elif use_rev:
            print('graph_in_dim reset as', graph_in_dim)
            rev_vectors = get_vector(reviews_dict, emb_size, rev_num, device, s_text_emb_method=s_t_emb_method)
            pre_vec_dict = {'review': rev_vectors}
            graph_in_dim = pre_vec_dict['review'].shape[1]
        else:
            pre_vec_dict = {}
            print('Do not use pre nlp info.')

        graph_dict = {('user', 'buy', 'item'): graph_info['ui_pair'],
                      ('item', 'buy-by', 'user'): graph_info['iu_pair']}
        if use_des and use_rev and self_loop:
            graph_dict[('user', 'wrt', 'review')] = graph_info['ur_pair']
            graph_dict[('review', 'wrt-by', 'user')] = graph_info['ru_pair']
            graph_dict[('review', 'rev', 'item')] = graph_info['ri_pair']
            graph_dict[('item', 'rev-by', 'review')] = graph_info['ir_pair']
            graph_dict[('description', 'des', 'item')] = graph_info['di_pair']
            graph_dict[('item', 'des-by', 'description')] = graph_info['id_pair']
            graph_dict[('user', 'u-self-loop', 'user')] = graph_info['uu_pair']
            graph_dict[('item', 'i-self-loop', 'item')] = graph_info['ii_pair']
            graph_dict[('description', 'd-self-loop', 'description')] = graph_info['dd_pair']
            graph_dict[('review', 'r-self-loop', 'review')] = graph_info['rr_pair']
        elif use_des and self_loop:
            graph_dict[('description', 'des', 'item')] = graph_info['di_pair']
            graph_dict[('item', 'des-by', 'description')] = graph_info['id_pair']
            graph_dict[('user', 'u-self-loop', 'user')] = graph_info['uu_pair']
            graph_dict[('item', 'i-self-loop', 'item')] = graph_info['ii_pair']
            graph_dict[('description', 'd-self-loop', 'description')] = graph_info['dd_pair']
        elif use_rev and self_loop:
            graph_dict[('user', 'wrt', 'review')] = graph_info['ur_pair']
            graph_dict[('review', 'wrt-by', 'user')] = graph_info['ru_pair']
            graph_dict[('review', 'rev', 'item')] = graph_info['ri_pair']
            graph_dict[('item', 'rev-by', 'review')] = graph_info['ir_pair']
            graph_dict[('user', 'u-self-loop', 'user')] = graph_info['uu_pair']
            graph_dict[('item', 'i-self-loop', 'item')] = graph_info['ii_pair']
            graph_dict[('review', 'r-self-loop', 'review')] = graph_info['rr_pair']
        elif use_des and use_rev:
            graph_dict[('user', 'wrt', 'review')] = graph_info['ur_pair']
            graph_dict[('review', 'wrt-by', 'user')] = graph_info['ru_pair']
            graph_dict[('review', 'rev', 'item')] = graph_info['ri_pair']
            graph_dict[('item', 'rev-by', 'review')] = graph_info['ir_pair']
            graph_dict[('description', 'des', 'item')] = graph_info['di_pair']
            graph_dict[('item', 'des-by', 'description')] = graph_info['id_pair']
        elif use_des:
            graph_dict[('description', 'des', 'item')] = graph_info['di_pair']
            graph_dict[('item', 'des-by', 'description')] = graph_info['id_pair']
        elif use_rev:
            graph_dict[('user', 'wrt', 'review')] = graph_info['ur_pair']
            graph_dict[('review', 'wrt-by', 'user')] = graph_info['ru_pair']
            graph_dict[('review', 'rev', 'item')] = graph_info['ri_pair']
            graph_dict[('item', 'rev-by', 'review')] = graph_info['ir_pair']
        elif self_loop:
            graph_dict[('user', 'u-self-loop', 'user')] = graph_info['uu_pair']
            graph_dict[('item', 'i-self-loop', 'item')] = graph_info['ii_pair']
        hetero_text_graph = dgl.heterograph(graph_dict, device=device)
        print(hetero_text_graph)

        model = HeteroGCNNet(hetero_text_graph, in_size=graph_in_dim, hidden_size=graph_hidden_dim,
                             out_size=graph_out_dim, use_dr_pre=dr_pre_train,
                             pre_v_dict=pre_vec_dict, if_non_active=if_non_active)
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-1)  # weight_decay: 1e-1

        # 构造uu ii矩阵
        ui_matrix = hetero_text_graph.adjacency_matrix(etype='buy').to_dense()
        iu_matrix = ui_matrix.transpose(0, 1)
        uu_con_matrix = torch.mm(ui_matrix, iu_matrix)
        ii_con_matrix = torch.mm(iu_matrix, ui_matrix)

        for i_e in range(epochs + 1):
            opt.zero_grad()
            model.forward(hetero_text_graph)

            b_num = args.batch_size // 2

            tuid = np.random.randint(user_num)
            pos_batch_users = get_x_pairs(uu_con_matrix, tuid, user_num, b_num, if_pos=True)
            neg_batch_users = get_x_pairs(uu_con_matrix, tuid, user_num, b_num, if_pos=False)

            p_u_value = torch.zeros(graph_out_dim).to(device)
            u_emb = model.lookup_emb('user', tuid)
            for (uid, p_uid, _) in pos_batch_users:
                p_u_emb = model.lookup_emb('user', p_uid)
                p_u_value += torch.pow(u_emb - p_u_emb, 2)
            n_u_value = torch.zeros(graph_out_dim).to(device)
            for (uid, n_uid) in neg_batch_users:
                n_u_emb = model.lookup_emb('user', n_uid)
                n_u_value += torch.pow(u_emb - n_u_emb, 2)
            u_loss = (p_u_value / b_num).sum() - (n_u_value / b_num).sum()
            # print('u_loss:', u_loss)

            tiid = np.random.randint(item_num)
            pos_batch_items = get_x_pairs(ii_con_matrix, tiid, item_num, b_num, if_pos=True)
            neg_batch_items = get_x_pairs(ii_con_matrix, tiid, item_num, b_num, if_pos=False)
            p_i_value = torch.zeros(graph_out_dim).to(device)
            i_emb = model.lookup_emb('item', tiid)
            for (iid, p_iid, _) in pos_batch_items:
                p_i_emb = model.lookup_emb('item', p_iid)
                p_i_value += torch.pow(i_emb - p_i_emb, 2)
            n_i_value = torch.zeros(graph_out_dim).to(device)
            for (iid, n_iid) in neg_batch_items:
                n_i_emb = model.lookup_emb('item', n_iid)
                n_i_value += torch.pow(i_emb - n_i_emb, 2)
            i_loss = (p_i_value / b_num).sum() - (n_i_value / b_num).sum()
            # print('i_loss:', i_loss)

            loss = u_loss + i_loss
            # print('loss:', loss)
            loss.backward()
            opt.step()

            if i_e % 10 == 0:
                str_loss = 'Epoch %3d: Loss %.4f' % (i_e, loss.item())
                print(str_loss)

        item_vectors = model.lookup_emb_list('item', need_all=True).cpu().detach().numpy()
        user_vectors = model.lookup_emb_list('user', need_all=True).cpu().detach().numpy()
    else:
        item_vectors = None
        user_vectors = None
        print('Not support this method.')
        exit(-1)

    np.save(save_root + '/' + data_name + '_embeddings.npy', item_vectors)
    np.save(save_root + '/' + data_name + '_user_embeddings.npy', user_vectors)

    # End time
    end_time = time.process_time()
    print("Cost time is %f" % (end_time - start_time))
