# coding: utf-8 -*- DQN model
# 修改版本 一次推荐一个 输出两个Q 前者代表Y(推荐)，后者代表N。
import tensorflow as tf
import numpy as np
import random
import math
import time
import os
from datetime import datetime
from util.dqn_env import RecommendENV

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#  DQN
#  hyper parameters
LR = 0.0001  # learning rate
GAMMA = 0.9  # reward discount
double_q = True  # if use Double DQN
e_greedy = 0.90
TAU = 0.01  # soft replacement


def xavier_init(in_num, out_num, constant=1):
    low = -constant * np.sqrt(6.0 / (in_num + out_num))
    high = constant * np.sqrt(6.0 / (in_num + out_num))
    return tf.random_uniform((in_num, out_num), minval=low, maxval=high, dtype=tf.float32)


class DQN(object):
    def __init__(self, sess, item_vector, s_dim, s_num, batch_size, MEMORY_CAPACITY,
                 reward_decay=GAMMA, e_greedy=e_greedy, e_greedy_increment=None,
                 learning_rate=LR, double_q=double_q):
        self.sess = sess
        self.item_vector = item_vector

        self.s_dim = s_dim
        self.s_num = s_num
        self.i_dim = self.s_dim

        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.keep_rate = 1

        self.memory_size = MEMORY_CAPACITY
        self.memory_counter = 0
        self.state_size = self.s_dim * self.s_num
        self.itme_size = self.i_dim
        self.memory = np.zeros((MEMORY_CAPACITY, self.state_size * 2 + self.itme_size * 2 + 2), dtype=np.float32)

        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        # 决定是否要使用double-q-network
        self.double_q = double_q
        self.learn_step_counter = 0
        self._build_net()

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.soft_replace_op = [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in zip(t_params, e_params)]

        sess.run(tf.global_variables_initializer())

    def _build_net(self):
        def build_layers(i_s, i_i, c_name):
            with tf.variable_scope('l1'):
                n_l1 = 100
                self.w1_s = tf.Variable(name='w1_s', initial_value=xavier_init(self.s_dim * self.s_num, n_l1),
                                        collections=c_name)
                self.w1_i = tf.Variable(name='w1_i', initial_value=xavier_init(self.i_dim, n_l1),
                                        collections=c_name)
                self.b1 = tf.Variable(name='b1', initial_value=tf.zeros([n_l1]))
                layer = tf.nn.relu(
                    tf.nn.dropout((tf.matmul(i_s, self.w1_s) + tf.matmul(i_i, self.w1_i) + self.b1),
                                  rate=1 - self.keep_rate))
            with tf.variable_scope('l2'):
                # Q(s,a)
                self.w2 = tf.Variable(name='w2', initial_value=xavier_init(n_l1, 2), collections=c_name)
                self.b2 = tf.Variable(name='b2', initial_value=tf.zeros([2]), collections=c_name)
                q_value = tf.matmul(layer, self.w2) + self.b2
            return q_value

        # input
        self.s = tf.placeholder(tf.float32, [None, self.s_num * self.s_dim], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.s_num * self.s_dim], name='s_')
        self.item = tf.placeholder(tf.float32, [None, self.i_dim], 'item')
        self.item_ = tf.placeholder(tf.float32, [None, self.i_dim], 'item_')
        self.q_target = tf.placeholder(tf.float32, [None, 2], name='Q-target')

        # evaluate_net
        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_eval = build_layers(self.s, self.item, c_names)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # target_net
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, self.item_, c_names)

    def store_transition(self, s, item, act_index, r, s_, item_):
        # print(s, item, act_index, r, s_, item_)
        transition = np.hstack((s, item, np.array(act_index).reshape((1, 1)),
                                np.array(r).reshape((1, 1)), s_, item_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, emb_s, emb_i):
        q_value = self.sess.run(self.q_eval, feed_dict={self.s: emb_s, self.item: emb_i})
        q_value = q_value.reshape(-1)
        # print(q_value)
        action = np.argmax(q_value)
        # print(self.epsilon)
        if np.random.random() > self.epsilon:
            action = np.random.randint(0, 2)
            # print(action)
        return action

    def learn(self):
        self.sess.run(self.soft_replace_op)

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run([self.q_next, self.q_eval],
                                            feed_dict={self.s: batch_memory[:, :self.state_size],
                                                       self.item: batch_memory[:, self.state_size:self.state_size + self.itme_size],
                                                       self.s_: batch_memory[:, -(self.itme_size + self.state_size):-self.itme_size],
                                                       self.item_: batch_memory[:, -self.itme_size:]})

        q_eval = self.sess.run(self.q_eval, feed_dict={
            self.s: batch_memory[:, :self.state_size],
            self.item: batch_memory[:, self.state_size:self.state_size + self.itme_size]})

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.state_size + self.itme_size]
        eval_act_index = np.array(eval_act_index, dtype=np.int32)
        reward = batch_memory[:, self.state_size + self.itme_size + 1]

        if self.double_q:
            # the action that brings the highest value is evaluated by q_eval
            max_act4next = np.argmax(q_eval4next, axis=1)
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next = np.max(q_next, axis=1)  # the natural DQN

        q_target = q_eval.copy()
        # print(batch_index, eval_act_index)
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, t_loss = self.sess.run([self.train_op, self.loss],
                                  feed_dict={self.s: batch_memory[:, :self.state_size],
                                             self.item: batch_memory[:, self.state_size:self.state_size + self.itme_size],
                                             self.q_target: q_target})

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        return t_loss

    def set_keep_rate(self, keep_rate):
        self.keep_rate = keep_rate


class RlProcess:
    def __init__(self, the_data_path, the_data_name, data_method, ste_method, epochs, state_num,
                 cand_size, one_u_steps, test_top_k: list, is_use_history=True):
        self.the_data_path = the_data_path
        self.the_data_name = the_data_name
        self.data_method = data_method
        self.ste_method = ste_method
        self.epochs = epochs
        self.state_num = state_num
        # Number of randomly selected candidate set items, make sure select_size < len(c_items_list)
        self.cand_size = cand_size
        self.one_u_steps = one_u_steps
        self.test_top_k = sorted(test_top_k)
        self.is_use_history = is_use_history
        self.item_vector = np.load(self.the_data_path + self.the_data_name + "_embeddings.npy")
        self.user_label_list, self.user_label_num = None, None
        self.cluster_items, self.cluster_users = None, None
        self.train_user_items_dict, self.train_user_items_rating_dict = None, None
        self.test_user_items_rating_dict = None
        self.nega_user_items_rating_dict = None
        self.supp_nega_cluster_items = None
        self.old_user2new, self.old_item2new = None, None
        self.data_shape, self.max_dis_dict = None, None
        self.test_dict = None
        # Data initialization
        self.data_prepare()

    # Data preparation
    def data_prepare(self):
        with open(self.the_data_path + "user_label_predict.txt", 'r') as l_file:
            self.user_label_list = l_file.read().split(',')  # str
        self.user_label_num = len(set(self.user_label_list)) - 1

        with open(self.the_data_path + 'train_user_items_dict.txt', 'r') as train_ui_dict:
            self.train_user_items_dict = eval(train_ui_dict.read())
        with open(self.the_data_path + 'train_user_items_rating_dict.txt', 'r') as train_uir_dict:
            self.train_user_items_rating_dict = eval(train_uir_dict.read())
        with open(self.the_data_path + 'test_user_items_rating_dict.txt', 'r') as test_uir_dict:
            self.test_user_items_rating_dict = eval(test_uir_dict.read())
        with open(self.the_data_path + 'nega_user_items_rating_dict.txt', 'r') as nega_uir_dict:
            self.nega_user_items_rating_dict = eval(nega_uir_dict.read())

        with open(self.the_data_path + 'cluster_users.txt', 'r') as c_us:
            self.cluster_users = eval(c_us.read())['cluster_users']

        with open(self.the_data_path + 'old_user2new.txt', 'r', encoding="utf-8") as f:
            self.old_user2new = eval(f.read())
        with open(self.the_data_path + 'old_item2new.txt', 'r', encoding="utf-8") as f:
            self.old_item2new = eval(f.read())
        self.data_shape = [len(self.old_user2new.keys()), len(self.old_item2new.keys())]

        # Obtain positive and negative samples for testing directly
        with open(self.the_data_path + 'test_dict.txt', 'r') as test_dict_file:
            self.test_dict = eval(test_dict_file.read())

        # Gets the list of classes that appear in the current class but not the farthest from the current class
        with open(self.the_data_path + 'supp_nega_cluster_items.txt', 'r') as nega_ci_file:
            self.supp_nega_cluster_items = eval(nega_ci_file.read())

    # hit_rate nDCG precision recall -- pre-user
    def result_evaluate(self, user_id: int, top_k_list: list, the_model, in_emb_s, batch_size):
        one_hit, one_ndcg, one_precision, one_recall = [], [], [], []
        h_test_items = self.test_dict[str(user_id) + '_p'].copy()
        test_candidate_items = h_test_items + self.test_dict[str(user_id) + '_n'].copy()
        # print('True Percent;', len(h_test_items) / len(test_candidate_items))
        random.shuffle(test_candidate_items)

        i_q_list = []
        round_num = int(len(test_candidate_items) / batch_size + 0.5)
        for ii in range(round_num):
            start = ii * batch_size
            end = (ii + 1) * batch_size
            if end > len(test_candidate_items):
                end = len(test_candidate_items)
            emb_item = np.zeros((end - start, self.item_vector.shape[1]))
            t_test_c_is = test_candidate_items[start: end]
            i = 0
            for test_item in t_test_c_is:
                emb_item[i] = self.item_vector[test_item]
                i += 1
            i_emb_s = np.tile(in_emb_s, [end - start, 1])
            q_values = the_model.sess.run(the_model.q_eval, {the_model.s: i_emb_s, the_model.item: emb_item})
            ii = 0
            for q_value in q_values:
                # print(q_value[0], q_value[1])
                if q_value[0] > q_value[1]:
                    i_q_list.append((t_test_c_is[ii], q_value[0]))
                ii += 1

        # print(len(i_q_list), i_q_list)
        i_q_list = sorted(i_q_list, reverse=True, key=lambda x: (x[1]))
        if len(i_q_list) > top_k_list[-1]:
            re_num = top_k_list[-1]
        else:
            re_num = len(i_q_list)
        recommend_list = []
        for re_i in range(re_num):
            recommend_list.append(i_q_list[re_i][0])
        # print(recommend_list)
        # print(test_item)
        # print(test_candidate_items)

        for top_k in top_k_list:
            hit_count = 0
            hit_list = []
            dcg = 0
            idcg = 0
            for k in range(len(recommend_list[:top_k])):
                t_item = recommend_list[k]
                if t_item in h_test_items:
                    hit_count += 1
                    t_rating = self.test_user_items_rating_dict[user_id][t_item] - 2
                    dcg += t_rating / math.log(k + 2)
                    hit_list.append(t_rating)
            hit_list.sort(reverse=True)
            # print(hit_list)
            kk = 0
            for t_rating in hit_list:
                idcg += t_rating / math.log(kk + 2)
                kk += 1
            if hit_count > 0:
                one_hit.append(1)
                one_ndcg.append(dcg / idcg)
                one_precision.append(hit_count / top_k)
                one_recall.append(hit_count / len(h_test_items))
            else:
                one_hit.append(0)
                one_ndcg.append(0)
                one_precision.append(0)
                one_recall.append(0)
        return one_hit, one_ndcg, one_precision, one_recall

    def runProcess(self):
        start_time = time.process_time()
        emb_size = self.item_vector.shape[1]
        s_dim, a_dim = emb_size, emb_size
        if not os.path.exists('reinforce_log/'):
            os.makedirs('reinforce_log/')
        reinforce_log = open('./reinforce_log/' + the_data_name + '_dqn_cluster-' + self.data_method
                             + '-' + self.ste_method
                             + '_cluster' + str(self.user_label_num + 1)
                             + '_state' + str(self.state_num) + '_'
                             + datetime.now().strftime("%Y%m%d_%H%M%S") + '.txt', 'w')

        BATCH_SIZE = 32
        MEMORY_CAPACITY = 1000
        boundry_rating = 2
        print('MEMORY_CAPACITY:', MEMORY_CAPACITY)
        env = RecommendENV(s_num=self.state_num, a_num=1, state_dim=s_dim,
                           item_vector=self.item_vector, supp_nega_cluster_items=self.supp_nega_cluster_items,
                           boundry_rating=boundry_rating, train_user_items_dict=self.train_user_items_dict,
                           train_user_items_rating_dict=self.train_user_items_rating_dict,
                           nega_user_items_rating_dict=self.nega_user_items_rating_dict,
                           user_label_list=self.user_label_list, data_shape=self.data_shape)

        # Control training times parameter setting
        MAX_STEPS = MEMORY_CAPACITY * 2  # Maximum training steps
        MIN_STEPS = MEMORY_CAPACITY * 1  # Minimum training steps, greater than or equal to memory
        once_show_num = 10
        # The convergence stop indicator
        # stops when the percentage of the sub average value to the original value is less than or equal to this value
        stop_line = 0.1

        # result_evaluate
        total_hits, total_ndcgs, total_precisions, total_recalls = [], [], [], []
        for _ in self.test_top_k:
            total_hits.append([])
            total_ndcgs.append([])
            total_precisions.append([])
            total_recalls.append([])
        cluster_w = []
        # t_sun = 0
        for i in range(self.user_label_num + 1):
            cluster_w.append(len(self.cluster_users[i]) / self.data_shape[0])
            # t_sun += cluster_w[i]
        # print(cluster_w, t_sun)

        total_cluster_steps = 0
        for i in range(self.user_label_num + 1):
            # user_cluster
            cluster_hits, cluster_ndcgs, cluster_precisions, cluster_recalls = [], [], [], []
            for _ in self.test_top_k:
                cluster_hits.append([])
                cluster_ndcgs.append([])
                cluster_precisions.append([])
                cluster_recalls.append([])

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as o_sess:
                # Each cluster corresponds to a graph
                dqn = DQN(o_sess, self.item_vector, s_dim, self.state_num,
                          BATCH_SIZE, MEMORY_CAPACITY)
                # Create a saver.
                cluster_saver = tf.train.Saver()
                the_saver_path = './reinforce_log/' + the_data_name + '_dqn_cluster' \
                                 + '/state' + str(self.state_num)  \
                                 + '/' + self.data_method + '-' + self.ste_method
                                 + '/c' + str(i)
                meta_path = the_saver_path + '/model.meta'
                if self.is_use_history:
                    if os.path.exists(meta_path) \
                            and os.path.exists(the_saver_path):
                        cluster_saver = tf.train.import_meta_graph(meta_path)
                        cluster_saver.restore(o_sess, tf.train.latest_checkpoint(the_saver_path))
                        print('Filled with', the_saver_path)
                user_size = len(self.cluster_users[i])  # Transboundary control

                # Initialize memory buffer
                for ii in range(MEMORY_CAPACITY):
                    user_id = self.cluster_users[i][int(ii % user_size)]

                    # Construct candidate set once per selection
                    train_num = int(positive_percent * self.cand_size)
                    if train_num > len(self.train_user_items_dict[user_id]):
                        train_num = len(self.train_user_items_dict[user_id])
                    train_list = random.sample(self.train_user_items_dict[user_id], train_num)
                    nega_num = int((self.cand_size - train_num) / 2)
                    nega_list = env.getNegative(user_id=user_id, nega_num=nega_num,
                                                supp_nega_cluster_items=self.supp_nega_cluster_items[
                                                    int(self.user_label_list[user_id])])
                    cand_set = train_list + nega_list
                    num_random = self.cand_size - train_num - len(nega_list)
                    random_list = env.getRandom(cand_set, num_random)
                    cand_set += random_list
                    random.shuffle(cand_set)

                    s, emb_s = env.init_env(user_id=user_id)
                    ii = 0
                    for item_index in cand_set:
                        emb_item = self.item_vector[item_index]
                        emb_item = emb_item.reshape((1, emb_item.shape[0]))
                        a_index = dqn.choose_action(emb_s, emb_item)
                        r, s_, emb_s_ = env.step_dqn(user_id=user_id,
                                                     in_state=s,
                                                     emb_s=emb_s,
                                                     index_a=a_index,
                                                     item_a_t=item_index,
                                                     train_list=train_list,
                                                     nega_list=nega_list)
                        if ii < len(cand_set) - 1:
                            emb_item_ = self.item_vector[cand_set[ii+1]]
                            emb_item_ = emb_item_.reshape((1, emb_item_.shape[0]))
                            dqn.store_transition(emb_s, emb_item, a_index, r, emb_s_, emb_item_)
                        ii += 1
                        s = s_
                        emb_s = emb_s_
                hit_list, ndcg_list, precision_list, recall_list = [], [], [], []
                for epoch in range(self.epochs):
                    str_cluster = 'Cluster:' + str(i)
                    str_td_loss = 'Loss:'
                    str_reward = 'Mean_Reward:'
                    t_sum_steps = 0
                    step_count = 1
                    once_show_r = 0
                    o_td_error = 0
                    tt_time_list = []
                    td_time_list = []

                    # training
                    dqn.set_keep_rate(keep_rate=0.8)
                    while True:
                        user_id = self.cluster_users[i][random.randint(0, user_size - 1)]
                        s, emb_s = env.init_env(user_id=user_id)
                        # A certain number of training for each user
                        # Construct candidate set once per selection
                        train_num = int(positive_percent * self.cand_size)
                        if train_num > len(self.train_user_items_dict[user_id]):
                            train_num = len(self.train_user_items_dict[user_id])
                        train_list = random.sample(self.train_user_items_dict[user_id], train_num)
                        nega_num = int((self.cand_size - train_num) / 2)
                        nega_list = env.getNegative(user_id=user_id, nega_num=nega_num,
                                                    supp_nega_cluster_items=self.supp_nega_cluster_items[
                                                        int(self.user_label_list[user_id])])
                        cand_set = train_list + nega_list
                        num_random = self.cand_size - train_num - len(nega_list)
                        random_list = env.getRandom(cand_set, num_random)
                        cand_set += random_list
                        random.shuffle(cand_set)

                        td_stime = time.process_time()
                        ii = 0
                        for item_index in cand_set:
                            emb_item = self.item_vector[item_index]
                            emb_item = emb_item.reshape((1, emb_item.shape[0]))
                            a_index = dqn.choose_action(emb_s, emb_item)
                            r, s_, emb_s_ = env.step_dqn(user_id=user_id,
                                                         in_state=s,
                                                         emb_s=emb_s,
                                                         index_a=a_index,
                                                         item_a_t=item_index,
                                                         train_list=train_list,
                                                         nega_list=nega_list)
                            if ii < len(cand_set) - 1:
                                emb_item_ = self.item_vector[cand_set[ii + 1]]
                                emb_item_ = emb_item_.reshape((1, emb_item_.shape[0]))
                                dqn.store_transition(emb_s, emb_item, a_index, r, emb_s_, emb_item_)
                            ii += 1
                            s = s_
                            emb_s = emb_s_
                            once_show_r += r
                        td_time_list.append(time.process_time() - td_stime)

                        # train
                        tt_stime = time.process_time()
                        o_td_error = dqn.learn()
                        tt_time_list.append(time.process_time() - tt_stime)
                        # print(o_td_error)

                        if step_count % once_show_num == 0:
                            # print('State_flag:', s_flag)
                            # print('State:', s)
                            new_loss = o_td_error / (once_show_num * self.one_u_steps)
                            str_td_loss += str(new_loss) + ' '
                            str_reward += str(once_show_r / (once_show_num * self.one_u_steps)) + ' '

                            if step_count >= MIN_STEPS:
                                # Take absolute value to prevent division by 0
                                if np.abs(old_loss - new_loss) / (np.abs(old_loss) + 0.000001) < stop_line:
                                    break

                            old_loss = new_loss
                            once_show_r = 0
                            o_td_error = 0
                        # print(t_td_error)
                        if step_count >= MAX_STEPS:
                            break
                        step_count += 1
                    log_time_cost = '\nTime cost for per training step:' + str(np.array(tt_time_list).mean() * 1000) + 'ms.' \
                                    + '\nTime cost for per decision:' + str(np.array(td_time_list).mean() * 1000) + 'ms.'
                    t_sum_steps += step_count
                    str_steps = 'Steps:' + str(t_sum_steps * self.one_u_steps)
                    str_train_log = str_cluster + '\n' + str_td_loss + '\n' + str_reward + '\n' \
                                    + str_steps + log_time_cost
                    print(str_train_log)
                    reinforce_log.write(str_train_log + '\n')
                    reinforce_log.flush()
                    total_cluster_steps += step_count

                    # Test and use the parameters of the corresponding class before changing the cluster
                    dqn.set_keep_rate(keep_rate=1)
                    for t_user_id in self.cluster_users[i]:
                        try:
                            self.test_dict[str(t_user_id) + '_p']
                        except KeyError:
                            continue
                        # Initialize test environment
                        s, emb_s = env.init_test_env(t_user_id)
                        # print(s)
                        # test
                        one_hit, one_ndcg, one_precision, one_recall = self.result_evaluate(
                            user_id=t_user_id,
                            top_k_list=self.test_top_k,
                            the_model=dqn,
                            in_emb_s=emb_s,
                            batch_size=BATCH_SIZE)
                        # print(t_user_id, one_hit, one_ndcg, one_precision, one_recall)
                        kk = 0
                        for _ in self.test_top_k:
                            cluster_hits[kk].append(one_hit[kk])
                            cluster_ndcgs[kk].append(one_ndcg[kk])
                            cluster_precisions[kk].append(one_precision[kk])
                            cluster_recalls[kk].append(one_recall[kk])
                            kk += 1
                    # print(len(cluster_hits))
                    str_rate = 'Evaluate of cluster ' + str(i) + ', Epoch ' + str(epoch)
                    kk = 0
                    hit_t, ndcg_t, precision_t, recall_t = [], [], [], []
                    for top_k in self.test_top_k:
                        if len(cluster_hits) > 0:
                            cluster_hit = np.array(cluster_hits[kk]).mean()
                            cluster_ndcg = np.array(cluster_ndcgs[kk]).mean()
                            cluster_precision = np.array(cluster_precisions[kk]).mean()
                            cluster_recall = np.array(cluster_recalls[kk]).mean()
                        else:
                            cluster_hit = 0
                            cluster_ndcg = 0
                            cluster_precision = 0
                            cluster_recall = 0
                        cluster_f1 = 2 * cluster_precision * cluster_recall / (
                                cluster_precision + cluster_recall + 0.000001)
                        hit_t.append(cluster_hit)
                        ndcg_t.append(cluster_ndcg)
                        precision_t.append(cluster_precision)
                        recall_t.append(cluster_recall)
                        str_rate += '\nTop ' + str(top_k) + \
                                    '. Hit_rate:' + str(cluster_hit) + \
                                    ' nDCG:' + str(cluster_ndcg) + \
                                    ' Precision:' + str(cluster_precision) + \
                                    ' Recall:' + str(cluster_recall) + \
                                    ' F1:' + str(cluster_f1)
                        kk += 1
                    hit_list.append(hit_t)
                    ndcg_list.append(ndcg_t)
                    precision_list.append(precision_t)
                    recall_list.append(recall_t)
                    print(str_rate)
                    reinforce_log.write(str_rate + '\n')
                    reinforce_log.flush()
                best_pos = 0
                for ii in range(1, len(hit_list)):
                    if hit_list[ii][0] > hit_list[best_pos][0]:
                        best_pos = ii
                kk = 0
                for _ in self.test_top_k:
                    total_hits[kk].append(hit_list[best_pos][kk] * cluster_w[i])
                    total_ndcgs[kk].append(ndcg_list[best_pos][kk] * cluster_w[i])
                    total_precisions[kk].append(precision_list[best_pos][kk] * cluster_w[i])
                    total_recalls[kk].append(recall_list[best_pos][kk] * cluster_w[i])
                    kk += 1
                # Save model each class has its own model
                if not os.path.exists(the_saver_path):
                    os.makedirs(the_saver_path)
                cluster_saver.save(o_sess, os.path.join(the_saver_path, 'model'))
            # Clear variables previously defined in the default graph
            tf.reset_default_graph()

        str_log = 'dqn_rec'
        kk = 0
        for top_k in self.test_top_k:
            total_hr, total_ndcg, total_precision, total_recall = 0, 0, 0, 0
            # print(total_hits)
            for i in range(self.user_label_num + 1):
                total_hr += total_hits[kk][i]
                total_ndcg += total_ndcgs[kk][i]
                total_precision += total_precisions[kk][i]
                total_recall += total_recalls[kk][i]
            total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall + 0.000001)
            str_log += '\nTTop ' + str(top_k) + \
                       '. HR:' + str(total_hr) + \
                       ' nDCG:' + str(total_ndcg) + \
                       ' Precision:' + str(total_precision) + \
                       ' Recall:' + str(total_recall) + \
                       ' F1:' + str(total_f1)
            kk += 1
        str_steps = 'Total train steps:' + str(total_cluster_steps * self.one_u_steps)
        end_time = time.process_time()
        str_time = "Cost time is %f" % (end_time - start_time)
        reinforce_log.write(str_log + '\n' + str_steps +
                            '\n' + str_time)
        reinforce_log.flush()
        reinforce_log.close()
        print(str_log + '\n' + str_steps + ' ' + str_time)


if __name__ == '__main__':
    # mf sa sg
    data_method = 'sg'
    # glove sbert
    s_t_emb_method = 'sbert'
    if data_method == 'mf':
        s_t_emb_method = ''
    # 'Digital_Music' 'Beauty' 'Clothing_Shoes_and_Jewelry'
    the_data_name = 'Digital_Music'
    state_num = 20  # Number of items in the status
    # The 'action_num' fixed, Yes(0) or No(1), for per item, no other choices
    cand_size = 50
    one_u_steps = 10  # Training times per user
    test_top_k = [10, 20]  # Top_k during test
    epochs = 3  # Number of training rounds
    positive_percent = 0.1  # alpha, positive items percent of the candidate set

    the_data_path = './Data/' + the_data_name + '/' + data_method + '/' + '/' + s_t_emb_method + '/'
    rl_model = RlProcess(the_data_path=the_data_path,
                         the_data_name=the_data_name,
                         ste_method=self.ste_method,
                         data_method=s_t_emb_method,
                         epochs=epochs,
                         state_num=state_num,
                         cand_size=cand_size,
                         one_u_steps=one_u_steps,
                         test_top_k=test_top_k,
                         is_use_history=False)
    rl_model.runProcess()
