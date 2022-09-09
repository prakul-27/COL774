import numpy as np
import argparse
import part4
import os

parser = argparse.ArgumentParser()
parser.add_argument('train_path',
	metavar='train_path',
	type=str,
	help='input train_path',
	default=os.getcwd(),
	nargs='?')
parser.add_argument('test_path',
	metavar='test_path',
	type=str,
	help='input test_path',
	default=os.getcwd(),
	nargs='?')
args = parser.parse_args()

train_path = args.train_path
test_path = args.test_path

train_path_X, train_path_Y = train_path+'/X.csv', train_path+'/Y.csv'
test_path_X = test_path+'/X.csv'
part4.run(train_path_X, train_path_Y, test_path_X)