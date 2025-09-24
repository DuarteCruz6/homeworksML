import math

"""
            1/(n-1) * sum( (x1i - medx1)*(x2i - medx2) )
pearson =             -----------
            sqrt (sum ( (x1i- medx1)^2 ) / (n-1)) * sqrt (sum ( (x2i- medx2)^2 ) / (n-1))
"""

def get_med(n:int, x:list) -> float:
    return sum(x)/n


def calculate_numerator(n:int, medx1:float, medx2:float , x1:list, x2:list) -> float:
    total = 0
    for i in range(n):
        x_one_i = x1[i]
        x_two_i = x2[i]
        
        total += (x_one_i - medx1)*(x_two_i - medx2)
        
    numerator = 1 / (n-1)
    numerator*= total
    return numerator

def calculate_denominator(n:int, medx1:float, medx2:float , x1:list, x2:list) -> float:
    total1 = 0
    for i in range(n):
        x_one_i = x1[i]
        total1 += (x_one_i - medx1)**2
    total1*= (1/(n-1))
        
    total2 = 0
    for i in range(n):
        x_two_i = x2[i]
        total2 += (x_two_i - medx2)**2
    total2*= (1/(n-1))
    
    denominator = math.sqrt(total1*total2)
    return denominator

def calculate_pearson(n:int, x1:list, x2:list) -> float:
    medx1 = get_med(n,x1)
    medx2 = get_med(n,x2)
    numerator = calculate_numerator(n,medx1,medx2,x1,x2)
    denominator = calculate_denominator(n,medx1,medx2,x1,x2)
    pearson = numerator/denominator
    return pearson