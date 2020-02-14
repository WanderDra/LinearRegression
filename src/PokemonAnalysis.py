import numpy as np
from Readrer import read_file
import math


def analysis():
    # table, row_num = read_file('D:\\PyProject\\LinearRegression\\LinearRegression\\data\\Pokemon.csv')
    table, row_num = read_file('C:\\Main\\GitHub\\Code\\LinearRegression\\data\\Pokemon.csv')

    table.pop(0)
    original_data = np.array(table)
    analyze_data = original_data[:, 4:12].astype(float)
    discriminant = original_data[:, -1]
    data_size = len(analyze_data[0])
    predict_discriminant = []
    correct_rate = []
    hit_rate = []
    right_in_false_rate = []
    false_in_false_rate = []
    right_in_right_rate = []
    false_in_right_rate = []
    for col in range(0, data_size):
        data = analyze_data[:, col]
        prepared_data = np.vstack([data, discriminant])
        print(prepared_data)
        t, m, r, h, rifr, fifr, rirr, firr = gradient_descend(1000, prepared_data, 0.01)
        predict = int(t * m)
        predict_discriminant.append(predict)
        correct_rate.append(r)
        hit_rate.append(h)
        right_in_false_rate.append(rifr)
        false_in_false_rate.append(fifr)
        right_in_right_rate.append(rirr)
        false_in_right_rate.append(firr)
        # break

    ig_list = []
    for col in range(0, data_size):
        x1_y1 = correct_rate[col] * 0.5
        x1_y2 = (1 - correct_rate[col]) * 0.5
        x2_y1 = (hit_rate[col] - correct_rate[col]) * 0.5
        x2_y2 = 0.5 - x2_y1
        h_y = - (x1_y1 + x2_y1) * math.log(x1_y1 + x2_y1) - (x1_y2 + x2_y2) * math.log(x1_y2 + x2_y2)
        h_y_x = (0.5 * (
                    - x1_y1 / (x1_y1 + x1_y2) * math.log(x1_y1 / (x1_y1 + x1_y2)) - x1_y2 / (x1_y1 + x1_y2) * math.log(
                x1_y2 / (x1_y1 + x1_y2)))
                 + 0.5 * (- x2_y1 / (x2_y1 + x2_y2) * math.log(x2_y1 / (x2_y1 + x2_y2)) - x2_y2 / (
                            x2_y1 + x2_y2) * math.log(x2_y2 / (x2_y1 + x2_y2))))
        ig = h_y - h_y_x
        ig_list.append(ig)

    sort_index = np.argsort(ig_list)[::-1]
    print(sort_index)

    for i in range(0, data_size):
        print(i, '-------------------------------------------------------------')
        print('predicted_discriminant = ', predict_discriminant[i])
        print('correct_rate = ', correct_rate[i])
        print('hit_rate = ', hit_rate[i])
        print('ig(y) = ', ig_list[i])

    def tree(discriminants, train_data):
        depth = len(discriminants)
        tree = []
        count = 0
        for d in range(0, depth):
            for n in range(0, 2 ** d):
                tree.append(discriminants[sort_index[d]])
        # final_decision = []
        # rate_list = [1]
        # for d in range(1, depth + 1):
        #     for n in range(0, 2 ** depth, 2):
        #         if d == 1:
        #             rate_list.append(rate_list[int(4 ** (d - 1) + n / 4 - 1)] * right_in_right_rate[sort_index[d - 1]])
        #             rate_list.append(rate_list[int(4 ** (d - 1) + n / 4 - 1)] * false_in_right_rate[sort_index[d - 1]])
        #         else:
        #             rate_list.append(rate_list[int(4 ** (d - 1) + n / 4)] * correct_rate[sort_index[d - 1]])
        #             rate_list.append(rate_list[int(4 ** (d - 1) + n / 4)] * right_in_false_rate[sort_index[d - 1]])
        # # print(rate_list[2 ** depth - 1:])
        # temp_rate = rate_list[2 ** depth - 1:]
        # print(temp_rate)
        # for i in range(0, len(temp_rate), 2):
        #     if temp_rate[i] >= temp_rate[i + 1]:
        #         final_decision.append('True')
        #     else:
        #         final_decision.append('False')
        #
        # print(final_decision)

        final_decision = [0, 0] * len(tree[2 ** (depth - 1) - 1:])
        for line in train_data:
            position = 0
            depth = 0
            for i in sort_index:
                if line[i] > tree[position]:
                    position += 2 ** depth - 1
                    print(position, '===============')
                if line[i] < tree[position]:
                    position += 2 ** depth
                depth += 1
                print(depth)
            final_position = position - 2 ** (depth - 1)
            final_decision[final_position] += 1

        print(final_decision)
        # print(tree[2 ** (depth - 1) - 1:], '==============')

        print(tree)
        return tree


    tree(predict_discriminant, analyze_data)




def gradient_descend(max_loop, ori_data, learning_rate):  # data[0] = data    data[1] = discriminant
    theta = 0
    data = np.array(ori_data[0].astype(float))
    discriminant = np.array(ori_data[1])
    target_goal = 0
    for d in discriminant:
        if d == 'True':
            target_goal += 1
    # print(target_goal)
    # false_rate = target_goal / (len(discriminant) - target_goal)

    max_data = 0
    for num in data:
        if num > max_data:
            max_data = num

    scaled_data = data / max_data
    d = np.ones(len(scaled_data)).T
    previous_miss = len(scaled_data)

    for i in range(0, max_loop):
        predict = []
        for data in scaled_data:
            if data >= theta:
                predict.append('True')
            else:
                predict.append('False')
        goal = 0
        right = 0
        wrong = 0
        false_in_false = 0
        false_in_right = 0
        right_in_false = 0
        right_in_right = 0
        total_false = 0
        total_right = 0
        for i in range(0, len(discriminant)):
            if predict[i] == 'True' and discriminant[i] == 'True':
                goal += 1
                right += 1
                right_in_right += 1
                total_right += 1
            if predict[i] == 'True' and discriminant[i] == 'False':
                # pass
                goal += 0.7
                wrong += 1
                right_in_false += 1
                total_false += 1
            if predict[i] == 'False' and discriminant[i] == 'False':
                false_in_false += 1
                total_false += 1
            if predict[i] == 'False' and discriminant[i] == 'True':
                false_in_right += 1
                total_right += 1

        if goal >= 0:
            loss = (goal - target_goal) / len(scaled_data)
        # print(goal)
        # print(loss)
        gradient = np.dot(loss * scaled_data, d) / len(scaled_data)
        theta = theta + learning_rate * gradient
        # break
    correct_rate = right / (right + wrong)
    hit_rate = right / target_goal
    right_in_false_rate = right_in_false / total_false
    false_in_false_rate = false_in_false / total_false
    right_in_right_rate = right_in_right / total_right
    false_in_right_rate = false_in_right / total_right

    return theta, max_data, correct_rate, hit_rate, right_in_false_rate, false_in_false_rate, right_in_right_rate, false_in_right_rate



analysis()
