import numpy as np
import time as t

scores = np.array([1, 2, 3, 4, 5, 6, 7])
boxes = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

start = t.time()
np.array_str(scores)
end = t.time()
print(end - start)

start = t.time()
np.array_str(boxes)
end = t.time()
print(end - start)

start = t.time()
np.array2string(scores)
end = t.time()
print(end-start)

start = t.time()
np.array2string(boxes)
end = t.time()
print(end-start)
