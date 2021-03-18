import numpy as np
import math
from util.utils import constant_sample


class RecommendENV:
    def __init__(self, s_num, a_num, state_dim: int, item_vector, boundry_rating,
                 train_user_items_dict, train_user_items_rating_dict, nega_user_items_rating_dict,
                 user_label_list, data_shape, supp_nega_cluster_items):
        self.state_num = s_num
        self.action_num = a_num
        self.state_dim = state_dim
        self.item_vector = item_vector
        self.boundry_rating = boundry_rating
        self.train_user_items_dict = train_user_items_dict
        self.train_user_items_rating_dict = train_user_items_rating_dict
        self.nega_user_items_rating_dict = nega_user_items_rating_dict
        self.supp_nega_cluster_items = supp_nega_cluster_items
        self.user_label_list = user_label_list
        self.data_shape = data_shape

        self.nega_ui_dic = {}
        for u_id in self.nega_user_items_rating_dict.keys():
            try:
                nega_items_list = list((self.nega_user_items_rating_dict[u_id]).keys())
            except KeyError:
                nega_items_list = []
            self.nega_ui_dic[u_id] = nega_items_list

    # 初始化环境状态
    def init_env(self, user_id: int):
        state_emd = np.zeros((1, self.state_num * self.state_dim))
        t_count = 0
        h_train_items = self.train_user_items_dict[user_id].copy()
        if len(h_train_items) >= self.state_num:
            # 随机选择
            t = []

            in_state = constant_sample(h_train_items, self.state_num)
            for i_item in in_state:
                state_emd[0][t_count * self.state_dim:(t_count + 1) * self.state_dim] = self.item_vector[i_item]
                t_count += 1
        else:
            in_state = h_train_items
            while len(in_state) < self.state_num:
                in_state.append(-1)
            for i_item in in_state:
                if i_item == -1:
                    state_emd[0][t_count * self.state_dim:(t_count + 1) * self.state_dim] = np.zeros(self.item_vector[0].shape)
                else:
                    state_emd[0][t_count * self.state_dim:(t_count + 1) * self.state_dim] = self.item_vector[i_item]
                t_count += 1
        return in_state, state_emd

    def init_random_env(self, h_test_items):
        state_emd = np.zeros((1, self.state_num * self.state_dim))
        t_count = 0
        test_num = int(len(h_test_items)/2)
        if test_num >= self.state_num:
            test_num = self.state_num
            in_test = constant_sample(h_test_items, test_num)
            in_state = in_test
        else:
            in_test = constant_sample(h_test_items, test_num)
            in_state = in_test + constant_sample(range(0, self.data_shape[1]), self.state_num - test_num)
        for i_item in in_state:
            state_emd[0][t_count * self.state_dim:(t_count + 1) * self.state_dim] = self.item_vector[i_item]
            t_count += 1
        return in_state, state_emd, in_test

    # 初始化测试时候用户的状态
    def init_test_env(self, user_id: int):
        state_emd = np.zeros((1, self.state_num * self.state_dim))
        t_count = 0
        h_train_items = self.train_user_items_dict[user_id].copy()
        if len(h_train_items) >= self.state_num:
            # 用户时间线上最后的状态
            in_state = h_train_items[len(h_train_items) - self.state_num:]
            for i_item in in_state:
                state_emd[0][t_count * self.state_dim:(t_count + 1) * self.state_dim] = self.item_vector[i_item]
                t_count += 1
        else:
            in_state = h_train_items
            while len(in_state) < self.state_num:
                in_state.append(-1)
            for i_item in in_state:
                if i_item == -1:
                    state_emd[0][t_count * self.state_dim:(t_count + 1) * self.state_dim] = np.zeros(self.item_vector[0].shape)
                else:
                    state_emd[0][t_count * self.state_dim:(t_count + 1) * self.state_dim] = self.item_vector[i_item]
                t_count += 1
        return in_state, state_emd

    def step_dqn(self, user_id, in_state, emb_s, index_a, item_a_t, train_list, nega_list):
        reward = 0
        if index_a == 0:
            in_state_ = in_state.copy()
            in_emb_s_ = np.zeros((1, self.state_num * self.state_dim))
            try:
                nega_items_list = self.nega_ui_dic[user_id]
            except KeyError:
                nega_items_list = []
            if item_a_t in train_list:
                reward = self.train_user_items_rating_dict[user_id][item_a_t] - self.boundry_rating
            elif item_a_t in nega_items_list:
                reward = self.nega_user_items_rating_dict[user_id][item_a_t] - self.boundry_rating - 1
            elif item_a_t in nega_list:
                reward = -0.5
            # 滑动窗口 从前往后替换 并且保证不重复
            if item_a_t not in in_state_:
                in_state_.pop()
                in_state_.insert(0, item_a_t)
            # 更新emb
            ii = 0
            for s_item_ in in_state_:
                in_emb_s_[0][ii * self.state_dim:(ii + 1) * self.state_dim] = self.item_vector[s_item_]
                ii += 1
            return reward, in_state_, in_emb_s_
        else:
            return reward, in_state, emb_s

    # 获取负样本
    def getNegative(self, user_id: int, nega_num: int, supp_nega_cluster_items):
        try:
            nega_items_list = self.nega_ui_dic[user_id].copy()
        except KeyError:
            nega_items_list = []

        if len(nega_items_list) > 0:
            if len(nega_items_list) >= nega_num:
                negative_list = constant_sample(nega_items_list, nega_num)
            else:
                negative_list = nega_items_list
                if nega_num - len(negative_list) >= len(supp_nega_cluster_items):
                    negative_list += list(supp_nega_cluster_items)
                else:
                    negative_list = constant_sample(supp_nega_cluster_items, nega_num - len(negative_list))
        else:
            if nega_num >= len(supp_nega_cluster_items):
                negative_list = list(supp_nega_cluster_items)
            else:
                negative_list = constant_sample(supp_nega_cluster_items, nega_num)
        return negative_list

    # 获取随机的
    def getRandom(self, exit_list, num_random):
        random_list = []
        while True:
            if len(random_list) == num_random:
                break
            one = np.random.randint(self.data_shape[1])
            if (one not in random_list) and \
                    (one not in exit_list):
                random_list.append(one)
        return random_list