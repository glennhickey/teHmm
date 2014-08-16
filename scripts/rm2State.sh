#!/bin/bash

#gah, doesn't work on mac!
#cat $1 | sed -e "s/non-LTR\|DNAt\|LINE\|SINE\|LTR\|DNA\|RC\|Unknown\|Other/TE/g"

cat $1 | sed -e "s/non-LTR/TE/g" | sed -e "s/DNAt/TE/g" |sed -e "s/LINE/TE/g" |sed -e "s/SINE/TE/g" |sed -e "s/LTR/TE/g" |sed -e "s/DNA/TE/g" |sed -e "s/RC/TE/g" |sed -e "s/Unknown/TE/g" |sed -e "s/Other/TE/g"

