import matplotlib.pyplot as plt

#data
x = [1, 2, 3]
h = [10, 8,11]
c = ['red', 'yellow', 'orange']

#bar plot
plt.bar(x, height = h, color = c)

plt.savefig('abc.jpg')
