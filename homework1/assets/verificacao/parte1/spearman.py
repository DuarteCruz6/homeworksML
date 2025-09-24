import pearson as p
"""
 ranks stuff and then uses pearson formula
"""

def get_ranked(n: int, x: list) -> list:
    indexed_x = list(enumerate(x)) # pairs values with their index
    indexed_x.sort(key=lambda pair: pair[1])

    ranks = [0] * n
    i = 0
    while i < n:
        # find ties
        value = indexed_x[i][1]
        j = i
        while j < n and indexed_x[j][1] == value:
            j += 1
            
        avg_rank = sum(range(i + 1, j + 1)) / (j - i)
        for k in range(i, j):
            original_index = indexed_x[k][0]
            ranks[original_index] = avg_rank
        i = j
    return ranks

def calculate_spearman(n: int, x1: list, x2: list) -> float:
    rank_x1 = get_ranked(n, x1)
    rank_x2 = get_ranked(n, x2)
    return p.calculate_pearson(n, rank_x1, rank_x2)
