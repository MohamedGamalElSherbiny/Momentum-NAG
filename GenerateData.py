import numpy as np

class GenerateData:

    def __init__(self, a, b, n=50, start=0, end=20):
        """
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
        """
        self.a = a
        self.b = b
        self.n = n
        self.start = start
        self.end = end

    def generate_random_data(self):
        """Generates a random data

        Returns
        -------

        Tuple:

        x (Input Data) : int
        y (Output Data) : int

        """
        x = np.linspace(self.start, self.end, self.n)
        return x, self.a * x + self.b