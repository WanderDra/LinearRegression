import math


def h_y():
    data = [[12.0/21.0, 9.0/21.0]]
    result = 0.0
    for pair in data:
        if pair[0] == 0:
            part1 = -pair[0]
        else:
            part1 = -pair[0] * math.log(pair[0], 2)
        if pair[1] == 0:
            part2 = -pair[1]
        else:
            part2 = -pair[1] * math.log(pair[1], 2)
        result += part1 + part2

    print(result)

def q_2_1():
    data1 = [[7.0/8.0, 1.0/8.0]]
    data2 = [[5.0/13.0, 8.0/13.0]]
    result1 = 0.0
    result2 = 0.0
    for pair in data1:
        if pair[0] == 0:
            part1 = -pair[0]
        else:
            part1 = -pair[0] * math.log(pair[0], 2)
        if pair[1] == 0:
            part2 = -pair[1]
        else:
            part2 = -pair[1] * math.log(pair[1], 2)
        result1 += part1 + part2

    for pair in data2:
        if pair[0] == 0:
            part1 = -pair[0]
        else:
            part1 = -pair[0] * math.log(pair[0], 2)
        if pair[1] == 0:
            part2 = -pair[1]
        else:
            part2 = -pair[1] * math.log(pair[1], 2)
        result2 += part1 + part2

    print((8.0/21.0) * result1 + (13.0/21.0) * result2)


def q_2_2():
    data1 = [[7.0/10.0, 3.0/10.0]]
    data2 = [[5.0/11.0, 6.0/11.0]]
    result1 = 0.0
    result2 = 0.0
    for pair in data1:
        if pair[0] == 0:
            part1 = -pair[0]
        else:
            part1 = -pair[0] * math.log(pair[0], 2)
        if pair[1] == 0:
            part2 = -pair[1]
        else:
            part2 = -pair[1] * math.log(pair[1], 2)
        result1 += part1 + part2

    for pair in data2:
        if pair[0] == 0:
            part1 = -pair[0]
        else:
            part1 = -pair[0] * math.log(pair[0], 2)
        if pair[1] == 0:
            part2 = -pair[1]
        else:
            part2 = -pair[1] * math.log(pair[1], 2)
        result2 += part1 + part2

    print((10.0/21.0) * result1 + (11.0/21.0) * result2)


h_y()
# q_2_1()
# q_2_2()
