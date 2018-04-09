import numpy as np

array = [["sd", 1, 2, 3, 4],
         ["hello", 5, 6, 7, 8]]
array = np.array(array)
for i in range(3):
    print(array[:, i])
