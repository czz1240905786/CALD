#!/bin/bash
int=10
while(( $int <=90 ))
do
	echo "budget $int"
	source to_shell.sh
	source cald.sh
	source to_shell.sh
	source clear.sh 
	python cald_train.py --gpu-id 0 --budget-num $int > ./experiment_cald/cald_result_budget$int.txt
	let "int+=4"
done
