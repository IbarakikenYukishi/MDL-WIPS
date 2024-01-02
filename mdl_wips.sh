#!/bin/bash
save_dir="results"
# save_dir="results_test"
webkb_path="data"
cora_path="data"
citeseer_path="data/citeseer"
amazon_path="data"
pubmed_path="data/pubmed"

device=0

# 引数の数を確認
if [ "$#" -ne 4 ]; then
  echo "# of arguments should be four, parameter_num, graph_type, task, and n_devices."
  exit 1
fi

parameter_num=${1}
graph_type=${2}
task=${3}
n_devices=${4}

hidden_size=1000

# other parameters
if [ $graph_type = "cora" ]; then
    iter=100000
    eval_each=1250
    init_lrs=(3.2 2.8 2.4 2.0 1.6 1.4 1.2 1.0 0.8 0.6 0.4)
    batchsize=64
elif [ $graph_type = "webkb" ]; then
    iter=100000
    eval_each=1250
    init_lrs=(3.2 2.8 2.4 2.0 1.6 1.4 1.2 1.0 0.8 0.6 0.4)
    # init_lrs=(1.6 0.8 0.4)
    batchsize=64
elif [ $graph_type = "citeseer" ]; then
    iter=100000
    eval_each=2500
    init_lrs=(3.2 2.8 2.4 2.0 1.6 1.4 1.2 1.0 0.8 0.6 0.4)
    # init_lrs=(1.6 0.8 0.4)
    batchsize=64
elif [ $graph_type = "pubmed" ]; then
    iter=300000
    eval_each=10000
    init_lrs=(3.2 1.6 0.8)
    batchsize=128
elif [ $graph_type = "amazon" ]; then
    iter=100000
    eval_each=1250
    init_lrs=(1.6 0.8 0.4)
    batchsize=64
fi

# MDL_WIPS
model_name="MDL_WIPS"
for init_lr in "${init_lrs[@]}"
do
    exp_name="comparison_${model_name}_${task}_${graph_type}_${parameter_num}_${init_lr}"
    command="conda activate embed; python ablation_study.py -amazon_path ${amazon_path} -task ${task} -exp_name ${exp_name} -save_dir ${save_dir} -model_name ${model_name} -cuda ${device} -graph_type ${graph_type} -webkb_path ${webkb_path} -cora_path ${cora_path} -citeseer_path ${citeseer_path} -pubmed_path ${pubmed_path} -iter ${iter} -eval_each ${eval_each} -init_lr ${init_lr} -batchsize ${batchsize} -parameter_num ${parameter_num} -hidden_size ${hidden_size}"
    echo "${command}"
    screen -dm bash -c "${command}"
    sleep 2
    device=$((device+1))
    device=$((device % n_devices))
done

