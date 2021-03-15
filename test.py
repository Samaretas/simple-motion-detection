import numpy as np

testarray = np.array([1,2,3,4,5,6])
arr2 = testarray.reshape((2,3))

# print(testarray)
print(arr2)
print("TESTS:")
# print(arr2[0][0])
# print(arr2[0][1])
# print(arr2[1][0])
# print(arr2[1][1])

# print((arr2-arr2))
# print(np.sum(arr2-arr2))

print(arr2[0:1, 1])