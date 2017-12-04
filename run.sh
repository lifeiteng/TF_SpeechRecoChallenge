#!/bin/bash

mode="train"
resnet_size=20
model="resnet15"  # resnet
suffix=
opt=

iter=""
infer_opt=""
test_dir="/home/ftli/Data/TF_Speech/test"
use_gpu=false

. ./parse_options.sh

set -e

model_dir=exp/speech_commands_${model}$suffix
datestr=`date "+DATE: %Y-%m-%d-%H-%M" | awk '{print $2}'`

if [ "$mode" = 'train' ];then
  export CUDA_VISIBLE_DEVICES=0
  python speech_commands/train.py --data_dir ~/Data/TF_Speech/speech_commands \
    --train_dir $model_dir --check_nans False \
    --how_many_training_steps 15000,15000,15000 --learning_rate 0.001,0.0001,0.00001 \
    --resnet_size $resnet_size --optimizer "$opt" \
    --summaries_dir $model_dir/train --model_architecture "$model"
elif [ "$mode" = 'train2' ];then
  # DEEP RESIDUAL LEARNING FOR SMALL-FOOTPRINT KEYWORD SPOTTING
  export CUDA_VISIBLE_DEVICES=0
  python speech_commands/train.py \
    --data_dir ~/Data/TF_Speech/speech_commands \
    --train_dir $model_dir \
    --how_many_training_steps 10000,10000,10000,10000 --learning_rate 0.001,0.0001,0.00005,0.00001 \
    --resnet_size $resnet_size --optimizer "$opt" \
    --summaries_dir $model_dir/summaries \
    --model_architecture "$model" --window_size_ms 30.0 --window_stride_ms 10.0 \
    --batch_size 64
fi

# [ ! -z "$mode" ] && datestr=$mode
[ -z "$iter" ] && iter=`cat $model_dir/checkpoint | grep "^model_checkpoint_path" | cut -d"\"" -f2`

step=`echo $iter | cut -d"-" -f2`
datestr="step-$step.$datestr"
echo "datestr = $datestr"

if $use_gpu;then
  export CUDA_VISIBLE_DEVICES=0
else
  export CUDA_VISIBLE_DEVICES=""
fi
echo "$0: $model_dir/$iter ======="

mkdir -p submissions
mkdir -p data

python speech_commands/infer.py \
  --data_dir $test_dir \
  --train_dir $model_dir \
  --resnet_size $resnet_size \
  --model_architecture "$model" --window_size_ms 30.0 --window_stride_ms 10.0 \
  --batch_size 64 --output_csv $model_dir/output_$datestr.csv </dev/null || exit 1

echo "$0: DONE"
