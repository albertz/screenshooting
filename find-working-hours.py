#!/usr/bin/python -u

from glob import glob
from itertools import *
from subprocess import *

# last time i checked in: Oct 29
# last working time: 27.9.10

files = []
files += glob("2010-09-30.*.png")
files += glob("2010-10-*.png")

Bin = "./find-eclipse-simple.py"
ProcessNum = 2
procs = []

def handleProc(proc):
	stdoutdata,_ = proc.communicate()
	print stdoutdata.strip()

def waitForFirstProc():
	global procs
	if len(procs) == 0: return	
	handleProc(procs[0])
	procs = procs[1:]

def waitForAllProcs():
	while len(procs) > 0: waitForFirstProc()

def spawnProcForFile(f):
	global procs
	procs += [Popen([Bin, f], stdout=PIPE)]

for f in files:
	if len(procs) >= ProcessNum: waitForFirstProc()
	spawnProcForFile(f)
	
waitForAllProcs()
