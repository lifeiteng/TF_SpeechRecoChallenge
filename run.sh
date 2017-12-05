#!/bin/bash

mode="train"
resnet_size=20
model="resnet15"  # resnet
hparams=""
batch_size=64

suffix=
opt=

iter=""
infer_opt=""
test_dir="/home/ftli/Data/TF_Speech/test"
use_gpu=true


training_steps="15000,10000"
learning_rate="0.001,0.0001"

. ./parse_options.sh

set -e

model_dir=exp/speech_commands_${model}$suffix
datestr=`date "+DATE: %Y-%m-%d-%H-%M" | awk '{print $2}'`

if $use_gpu;then
  export CUDA_VISIBLE_DEVICES=0
else
  export CUDA_VISIBLE_DEVICES=""
fi

if [ "$mode" = 'train' ];then
  python speech_commands/train.py \
    --data_dir ~/Data/TF_Speech/speech_commands \
    --train_dir $model_dir \
    --how_many_training_steps "$training_steps" --learning_rate "$learning_rate" \
    --resnet_size $resnet_size --optimizer "$opt" --hparams "$hparams" \
    --summaries_dir $model_dir/summaries \
    --model_architecture "$model" --window_size_ms 30.0 --window_stride_ms 10.0 \
    --batch_size $batch_size
fi

# [ ! -z "$mode" ] && datestr=$mode
[ -z "$iter" ] && iter=`cat $model_dir/checkpoint | grep "^model_checkpoint_path" | cut -d"\"" -f2`

step=`echo $iter | cut -d"-" -f2`
datestr="step-$step.$datestr"
echo "datestr = $datestr"

echo "$0: $model_dir/$iter ======="

mkdir -p submissions
mkdir -p data

python speech_commands/infer.py \
  --data_dir $test_dir \
  --train_dir $model_dir \
  --resnet_size $resnet_size --hparams "$hparams" \
  --model_architecture "$model" --window_size_ms 30.0 --window_stride_ms 10.0 \
  --batch_size 64 --output_csv $model_dir/output_$datestr.csv </dev/null || exit 1

cp $model_dir/output_$datestr.csv submissions || exit 1

echo "$0: DONE"
