import numpy
import matplotlib.pyplot as pyplot
import math
import pandas


# Pre-defined parameters
mean = 0.0
sigma = 1.0
delta_t = 1.0/250.0

BM1 = numpy.append(numpy.array([0]), numpy.random.normal(mean, sigma, 250))
BM1 = BM1*math.sqrt(delta_t)
print("BM1")
print("sample mean:     %.8f"%numpy.mean(BM1))
print("sample variance: %.8f\n"%numpy.std(BM1, ddof=1)**2)
BM1 = numpy.cumsum(BM1)

BM2 = numpy.append(numpy.array([0]), numpy.random.normal(mean, sigma, 250))
BM2 = BM2*math.sqrt(delta_t)
print("BM2")
print("sample mean:     %.8f"%numpy.mean(BM2))
print("sample variance: %.8f\n"%numpy.std(BM2, ddof=1)**2)
BM2 = numpy.cumsum(BM2)

BM3 = numpy.append(numpy.array([0]), numpy.random.normal(mean, sigma, 250))
BM3 = BM3*math.sqrt(delta_t)
print("BM3")
print("sample mean:     %.8f"%numpy.mean(BM3))
print("sample variance: %.8f\n"%numpy.std(BM3, ddof=1)**2)
BM3 = numpy.cumsum(BM3)


df = pandas.DataFrame({'BM1' : BM1, 'BM2' : BM2, 'BM3' : BM3})
with open("BM.csv", "w+") as fptr:
    df.to_csv(fptr, index=False)