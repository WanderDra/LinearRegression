import numpy as np
from Readrer import read_file
import math
import os
from matplotlib import pyplot as plt


def analysis():
    paths = os.path.abspath(os.path.dirname(__file__)).split('shippingSchedule')[0].split('\\')[:-1]
    project_dir = paths[0]
    paths.pop(0)
    for path in paths:
        project_dir += os.sep + path
    table, row_num = read_file(
        os.path.join(project_dir, 'data', 'Pokemon.csv'))


    # Decide discriminant by gradient descend
    title = table.pop(0)
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

    # Get IG(Y) and build the tree
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

    # Train the tree
    def train(discriminants, train_data, dis_list):
        depth = len(discriminants)
        tree = []
        count = 0
        for d in range(0, depth):
            for n in range(0, 2 ** d):
                tree.append(discriminants[sort_index[d]])
        print(tree)

        final_decision = [0, 0, 0, 0] * len(tree[2 ** (depth - 1) - 1:])
        counter = 0
        for line in train_data:
            position = 0
            depth = 1
            for i in sort_index:
                # print('line: ', i, ' ', line[i])
                # print('tree: ', position, ' ', tree[position])
                if line[i] >= tree[position]:
                    position = 2 * position + 1
                elif line[i] < tree[position]:
                    position = 2 * position + 2
                # print('==================')
                depth += 1
            final_position = position - 2 ** (depth - 1) - 1
            if dis_list[counter] == 'True':
                final_decision[final_position * 2] += 1
            else:
                final_decision[final_position * 2 + 1] += 1
            counter += 1

        print(final_decision)
        final_output = []
        possibility = []
        for i in range(0, len(final_decision), 2):
            if final_decision[i] >= final_decision[i + 1]:
                final_output.append('True')
                if final_decision[i] == 0 and final_decision[i + 1] == 0:
                    possibility.append(None)
                else:
                    possibility.append(final_decision[i]/(final_decision[i] + final_decision[i + 1]))
            else:
                final_output.append('False')
                if final_decision[i] == 0 and final_decision[i + 1] == 0:
                    possibility.append(None)
                else:
                    possibility.append(final_decision[i + 1] / (final_decision[i] + final_decision[i + 1]))

        return tree, final_output, possibility

    # Train the tree
    train_data = analyze_data[0:650]
    tree, final_output, possibility = train(predict_discriminant, train_data, discriminant)
    print(final_output)

    depth = len(predict_discriminant) + 1
    def draw(depth):
        plt.figure()
        p = []
        d = 0
        for i in range(0, len(tree)):
            if i > (2 ** (d + 1)) - 2:
                d += 1
            y = 20.0 * (depth - d)
            pre_x = 0
            for h in range(d + 1, depth):
                pre_x = pre_x + 5 * (2 ** (depth - h - 1))
            x = 10.0 + pre_x + (i - (2 ** d) + 1) * (10 * (2 ** (depth - d - 1)))
            # print(x)
            p.append([x, y])

        print(p)
        for point in p:
            plt.plot(point[0], point[1], 'b.')
        for i in range(0, len(tree) - 2 ** (depth - 2)):
            # print(p[i][0])
            x1 = [p[i][0], p[2 * i + 1][0]]
            y1 = [p[i][1], p[2 * i + 1][1]]
            x2 = [p[i][0], p[2 * i + 2][0]]
            y2 = [p[i][1], p[2 * i + 2][1]]
            plt.plot(x1, y1, 'b-')
            plt.plot(x2, y2, 'b-')

        d = 0
        for i in range(0, len(tree)):
            if i > (2 ** (d + 1)) - 2:
                d += 1
            plt.text(p[i][0], p[i][1], str(title[4 + d]) + '>' + str(tree[i]) + '?')
        plt.show()

    draw(depth)




    # Put data in tree
    right = 0
    count = 0
    for i in range(651, len(analyze_data)):
        line = analyze_data[i]

        position = 0
        depth = 1
        for i in sort_index:
            # print('line: ', i, ' ', line[i])
            # print('tree: ', position, ' ', tree[position])
            if line[i] >= tree[position]:
                position = 2 * position + 1
            elif line[i] < tree[position]:
                position = 2 * position + 2
            # print('==================')
            depth += 1
        final_position = position - 2 ** (depth - 1) - 1
        print('----------------------------------------------')
        print(final_output[final_position], ' ', possibility[final_position])
        print('act = ', discriminant[i])
        print('----------------------------------------------')
        if final_output[final_position] == discriminant[i]:
            right += 1

        count += 1

    accuracy = right / count
    print('accuracy = ', accuracy)




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
                goal += 0.7             # 0.7 has the best performance via trying
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
