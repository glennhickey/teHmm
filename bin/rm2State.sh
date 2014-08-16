#!/bin/bash

cat $1 | sed -e "s/non-LTR\|DNAt\|LINE\|SINE\|LTR\|DNA\|RC\|Unknown\|Other/TE/g"

