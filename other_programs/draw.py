import matplotlib.pyplot as plt

aPiex = [862, -154]
aGps = [1447, 1558]
# cPiex = [-696, 14]
# cGps = [2394, -697]
cPiex = [696, -14]
cGps = [-2394, 697]
b = [0, 0]


def line(a,sig):
    x = [0]
    y = [0]
    x.append(a[0])
    y.append(a[1])
    plt.plot(x, y,sig)


line(aGps,'r-')
line(aPiex,'r--')
line(cGps,'b-')
line(cPiex,'b--')
plt.show()
