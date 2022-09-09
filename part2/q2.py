import numpy as np
import argparse
import part2
import os

parser = argparse.ArgumentParser()
parser.add_argument('test_path',
	metavar='test_path',
	type=str,
	help='input test_path',
	default=os.getcwd(),
	nargs='?')
args = parser.parse_args()

test_path = args.test_path
test_path_X = test_path+'/X.csv'
part2.run(train_path_X, train_path_Y, test_path_X)