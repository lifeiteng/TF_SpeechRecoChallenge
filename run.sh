#!/bin/bash

mode="train"
resnet_size=20
model="resnet15"  # resnet
hparams=""
batch_size=64
feature_scaling=''

suffix=
opt=

iter=""
infer_opt=""

data_dir=~/Data/TF_Speech/speech_commands
test_dir="/home/ftli/Data/TF_Speech/test"
use_gpu=true
train_opts=""
wanted_words="yes,no,up,down,left,right,on,off,stop,go"

skip_infer=false

score_prefix=""

training_steps="15000,10000"
learning_rate="0.001,0.0001"

. ./parse_options.sh

set -e

model_dir=exp/speech_commands_${model}$suffix

if $use_gpu;then
  export CUDA_VISIBLE_DEVICES=0
else
  export CUDA_VISIBLE_DEVICES=""
fi

if [ "$mode" = 'train' ];then
  python speech/train.py \
    --data_dir $data_dir \
    --train_dir $model_dir \
    --how_many_training_steps "$training_steps" --learning_rate "$learning_rate" \
    --optimizer "$opt" --hparams "$hparams" \
    --summaries_dir $model_dir/summaries \
    --model_architecture "$model" \
    --wanted_words "$wanted_words" \
    --batch_size $batch_size --feature_scaling "$feature_scaling" $train_opts
  if $skip_infer;then
    echo "$0: skip infer."
    exit 0
  fi
fi

datestr=`date "+DATE: %Y-%m-%d-%H-%M" | awk '{print $2}'`
# [ ! -z "$mode" ] && datestr=$mode
[ -z "$iter" ] && iter=`cat $model_dir/checkpoint | grep "^model_checkpoint_path" | cut -d"\"" -f2`

step=`echo $iter | cut -d"-" -f2`
datestr="step-$step.$datestr"
echo "datestr = $datestr"

echo "$0: $model_dir/$iter ======="

mkdir -p submissions
mkdir -p data

output_csv=$model_dir/scores${score_prefix}_${model}${suffix}_$datestr.csv

python speech/infer.py \
  --data_dir $test_dir \
  --train_dir $model_dir --hparams "$hparams" \
  --model_architecture "$model" \
  --batch_size 48 --output_csv $output_csv \
  --wanted_words "$wanted_words" \
  --feature_scaling "$feature_scaling" $train_opts </dev/null || exit 1

cp $output_csv submissions || exit 1
python speech/ensemble_label.py $output_csv submissions/${model}${suffix}_${score_prefix}$datestr.csv
python speech/ensemble_label.py --tune $output_csv submissions/${model}${suffix}_${score_prefix}${datestr}_tuned.csv

echo "$0: DONE"
