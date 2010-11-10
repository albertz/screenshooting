#!/bin/bash

opencv_createsamples -img eclipse-icon.png -vec eclipse-icon.vec \
	-bg eclipse-icon.negative -bgcolor 255 \
	-maxxangle 0 -maxyangle 0 -maxzangle 0 \
	-w 32 -h 32

rm -r eclipse-icon

opencv_haartraining -data eclipse-icon -vec eclipse-icon.vec \
	-bg eclipse-icon.negative \
	-maxfalsealarm 0.2 \
	-w 32 -h 32
