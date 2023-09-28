#!/bin/bash
# echo "dataset: ${1}, true_dim: ${2}"
save_dir="results"
webkb_path="data"
cora_path="data"
citeseer_path="data/citeseer"
pubmed_path="data/pubmed"

# datasets
# graph_type="webkb"
# graph_type="cora"
# graph_type="hierarchy"
# graph_type="pubmed"
# graph_type="citeseer"
# task="linkpred"

graph_type=${1}
task=${2}

if [ $graph_type = "cora" ]; then
    iter=100000
    eval_each=2500
    init_lrs=(0.8 0.4 0.2 0.1 0.05)
    batchsize=64
    # parameter_nums=(20 30 40 60 80 100)
    parameter_nums=(150 200 250 300)
    hidden_size=1000
elif [ $graph_type = "webkb" ]; then
    iter=100000
    eval_each=2500
    init_lrs=(0.8 0.4 0.2 0.1 0.05)
    batchsize=64
    # parameter_nums=(20 30 40 60 80 100)
    parameter_nums=(150 200 250 300)
    hidden_size=1000
elif [ $graph_type = "citeseer" ]; then
    iter=100000
    eval_each=2500
    init_lrs=(0.8 0.4 0.2 0.1 0.05)
    batchsize=64
    # parameter_nums=(20 30 40 60 80 100)
    parameter_nums=(150 200 250 300)
    hidden_size=1000
elif [ $graph_type = "pubmed" ]; then
    # iter=100000
    # eval_each=2500
    # init_lrs=(0.8 0.4 0.2 0.1 0.05)
    # batchsize=64
    # parameter_nums=(20 30 40 60 80 100)
    # hidden_size=1000
    iter=100000
    eval_each=5000
    init_lrs=(3.2 1.6 0.8 0.4 0.2)
    batchsize=128
    # parameter_nums=(20 30 40 60 80 100)
    parameter_nums=(150 200 250 300)
    hidden_size=1000
fi

device=0
n_devices=4

for init_lr in "${init_lrs[@]}"
do
    for parameter_num in "${parameter_nums[@]}"
    do
    	exp_name="${graph_type}_${parameter_num}_${init_lr}"
        command="conda activate embed; python ablation_study.py -task ${task} -exp_name ${exp_name} -save_dir ${save_dir} -model_name MDL_WIPS -cuda ${device} -graph_type ${graph_type} -webkb_path ${webkb_path} -cora_path ${cora_path} -citeseer_path ${citeseer_path} -pubmed_path ${pubmed_path} -iter ${iter} -eval_each ${eval_each} -init_lr ${init_lr} -batchsize ${batchsize} -parameter_num ${parameter_num} -hidden_size ${hidden_size}"
        echo "${command}"
        screen -dm bash -c "${command}"
        sleep 2
        device=$((device+1))
        device=$((device % n_devices))
    done
done