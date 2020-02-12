import math


def main():
    data = [[3.0/21.0, 0.0], [4.0/21.0, 1.0/21.0], [4.0/21.0, 3.0/21.0], [1.0/21.0, 5.0/21.0]]
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

def q_2():
    data1 = [[49.0/100, 51.0/100]]
    data2 = [[24.0/25.0, 1.0/25.0], [25.0/75.0, 50.0/75.0]]
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

    print(result1 - result2)


# main()
q_2()