#!/bin/bash
# echo "dataset: ${1}, true_dim: ${2}"
save_dir="results"
graph_type="webkb"
webkb_path="data"
iter=100000
eval_each=5000
# iter=10
# eval_each=5
init_lrs=(0.8 0.4 0.2 0.1 0.05)
# init_lrs=(1.0 0.1)
batchsize=64
# parameter_nums=(10 20 30 40 50 60 70 80 90 100 150 200 250 300)
parameter_nums=(10 20 30 40 50 60 70 80 90)
# parameter_nums=(100 150 200 250 300)
hidden_size=1000
device=0
n_devices=4

for init_lr in "${init_lrs[@]}"
do
    for parameter_num in "${parameter_nums[@]}"
    do
    	exp_name="${graph_type}_${parameter_num}_${init_lr}"
        command="conda activate embed; python ablation_study.py -task linkpred -exp_name ${exp_name} -save_dir ${save_dir} -model_name MDL_WIPS -cuda ${device} -graph_type ${graph_type} -webkb_path ${webkb_path} -iter ${iter} -eval_each ${eval_each} -init_lr ${init_lr} -batchsize ${batchsize} -parameter_num ${parameter_num} -hidden_size ${hidden_size}"
        echo "${command}"
        screen -dm bash -c "${command}"
        sleep 2
        device=$((device+1))
        device=$((device % n_devices))
    done
done