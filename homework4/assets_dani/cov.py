from math import pi, e

def cov_matrix(prob, points, centroid):

    res = []
    for i in range(2):
        aux = []
        for j in range(2):

            num = 0
            den = 0

            for n in range(len(prob)):

                num += prob[n] * (points[n][i] - centroid[i]) * (points[n][j] - centroid[j])
                den += prob[n]

            aux.append(num / den)

        res.append(aux)

    print(res)

prob1 = 0.6
prob2 = (0.3 / (pi * e)) / ((0.3 / (pi * e)) + (0.2 / ((e ** 5) * pi)))
prob3 = (0.3 / (pi * (e**2.5))) / ((0.3 / (pi * (e**2.5))) + (0.2 / ((e ** 8.5) * pi)))

prob = [1-prob1, 1-prob2, 1-prob3]
points = [(0,0),(2,0),(0,3)]
centroid = (0.05831643, 0.01196319)

cov_matrix(prob, points, centroid)