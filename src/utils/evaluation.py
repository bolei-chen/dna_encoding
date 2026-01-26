import math

def info_density(cardinality: int, codeword_length: int) -> float:
    """
    Calculate the information density of a codeword.
    The information density is the ratio of the number of codewords to the number of possible codewords.
    """
    return math.log2(cardinality) / codeword_length