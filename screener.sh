#!/bin/bash

t=10
#s="foo.png"
s=""

while true; do
	f="$(date "+%Y-%m-%d%.%H.%M.%S").png"
	echo "$f"
	screencapture -x $s "$f"
	[ "$s" != "" ] && rm $s
	sleep "$t" || break
done

