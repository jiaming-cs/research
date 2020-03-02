import numpy as np
import matplotlib.pyplot as plt

data = '''Average accuracy for normal surf: 0.687368
Average accuracy for hsv surf: 0.72
Average accuracy for normal surf: 0.697895
Average accuracy for hsv surf: 0.770526
Average accuracy for normal surf: 0.719649
Average accuracy for hsv surf: 0.784912
Average accuracy for normal surf: 0.740526
Average accuracy for hsv surf: 0.784737
Average accuracy for normal surf: 0.745053
Average accuracy for hsv surf: 0.789053
Average accuracy for normal surf: 0.759649
Average accuracy for hsv surf: 0.79807
Average accuracy for normal surf: 0.766466
Average accuracy for hsv surf: 0.799098
Average accuracy for normal surf: 0.766711
Average accuracy for hsv surf: 0.798421
Average accuracy for normal surf: 0.769708
Average accuracy for hsv surf: 0.797427
Average accuracy for normal surf: 0.771263
Average accuracy for hsv surf: 0.796526
Average accuracy for normal surf: 0.768708
Average accuracy for hsv surf: 0.798182'''

'''
num_str = []
for line in data.split("\n"):
    num_str.append(line.split(": ")[1])
'''

accuracy_64 = [0.646154, 0.667308, 0.697436, 0.750962, 0.780769, 0.810256, 0.824176, 0.829808, 0.838889, 0.837308, 0.831818]
accuracy_128 = [0.769231, 0.784615, 0.794872, 0.797115, 0.815385, 0.829487, 0.836264, 0.835096, 0.837179, 0.832692, 0.833916]
'''
for i in range(len(num_str)):
    if i % 2 == 0:
        accuracy_normal.append(float(num_str[i]))
    else:
        accuracy_knn.append(float(num_str[i]))
'''

x_axis = np.linspace(0.5, 1.5, 11)



mean_64 = sum(accuracy_64) / len(accuracy_64)
mean_128 = sum(accuracy_128) / len(accuracy_128)

print (mean_64)
print (mean_128)
plt.title("Average Accuracy of SURF-64 and SURF-128")
plt.plot(x_axis, accuracy_64, "o-", color = "red", label = "SURF-64")
plt.plot(x_axis, accuracy_128,"o-", color = "green", label = "SURF-128")
plt.legend()
plt.xlabel("Zoom Factor")
plt.ylabel("Accuracy")
plt.show()