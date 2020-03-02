import matplotlib.pyplot as plt

p = [203, 1, 8, 9, 12, 6, 0, 70, 21, 18, 13, 1, 0, 0, 33, 0, 2, 10, 6,\
     0, 4, 14, 17, 5, 0, 8, 7, 18, 9, 12, 18, 19, 23, 56, 306, 149, 94, 54, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

data = []

for i in range(len(p)):
    for x in range(p[i]):
        data.append(i*6)
plt.hist(data, 60, normed=True)
plt.title("Color Frequency Histogram of the Circle)")
plt.xlabel("Hue Value")
plt.ylabel("Frequency")
plt.show()
    