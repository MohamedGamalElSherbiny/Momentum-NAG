import numpy as np

def generate_random_data(a, b, n=50, start=0, end=20):
    """Generates a random data

    Parameters
    ----------

    a :         int
                Coefficient of x

    b :         int
                The constant term

    n :         int, optional
                The number of points

    start :     int, optional
                The starting point of x

    end :       int, optional
                The end point of x

    Returns
    -------

    Tuple:

    x (Input Data) : int
    y (Output Data) : int

    """
    x = np.linspace(start, end, n)
    return x, a * x + b

class GenerateData:
    pass