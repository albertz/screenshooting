#!/usr/bin/python

import re
from collections import *

#hours = OrderedDict()
hours = dict()

for l in open("working-hours.out").readlines():
	l = l.strip()
	m = re.match("^(....)-(..)-(..)\.(..)\.(..)\.(..)\.png : (.*)$", l)
	if not m:
		#print "error:", l
		continue
	#else:
	#	print m.groups()
	year,month,day,hour,minute,second,result = m.groups()
	day = (year,month,day)
	time = (hour,minute)
	if result.startswith("yes"):
		if not day in hours: hours[day] = set()
		hours[day].add(time)

days = sorted(list(hours))
for h in days:
	minutes = len(hours[h])
	print h, ":", str(minutes / 60) + ":" + str(minutes % 60)

