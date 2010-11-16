#!/usr/bin/python

from glob import glob
import random
dock = __import__("find-dock")

files = glob("*.png")
#files = glob("2010-10-*.png")
#files = glob("2010-10-11.*.png") # bottom dock with eclipse
#files = glob("2010-10-28.*.png") # left dock with eclipse
random.shuffle(files)

for f in files:
	print f, ":",
	print dock.getDockIcons(f)
	