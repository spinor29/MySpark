import sys
import numpy as np

from pyspark import SparkConf, SparkContext

def sparseRating(line):
	return int(line[0]), int(line[1]), float(line[2])

if __name__ == "__main__":

	conf = SparkConf().setAppName("Recommender Project")
	sc = SparkContext(conf=conf)

	data = sc.textFile("test.data")

	ratings = data.map(lambda l: l.split(',')).map(sparseRating) # a list of (user id, product id, rating)
    
