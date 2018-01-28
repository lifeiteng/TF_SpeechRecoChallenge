

data_dir=~/Data/TF_Speech/speech_commands
test_dir="/home/ftli/Data/TF_Speech/test"
batch_size=48

# conv 0.77
bash run.sh --mode train --model conv \
	--data_dir $data_dir --test_dir $test_dir \
  --hparams ""  --train-opts "" \
  --suffix "" --batch_size $batch_size \
  --opt "{ name: Adam, params: {} }" \
  --training_steps  "25000,25000" --skip-infer false \
  --learning_rate "0.01,0.001" --use-gpu true --feature_scaling ''


# resenet 0.89
dropout=0.8
hparams="resnet_filters=45,resnet_type=c,add_batch_norm=True,add_first_batch_norm=False,freeze_first_batch_norm=False"
suffix="15_filters45_BatchNormTypeC_CMN_Dropout${dropout}_Fbank80"

# cmvn
bash run.sh --mode train --model resnet \
	--data_dir $data_dir --test_dir $test_dir \
  --hparams "$hparams"  --train-opts "--dropout_prob $dropout --dct_coefficient_count 80 --feature_type fbank" \
  --suffix "$suffix" --batch_size $batch_size \
  --opt "{ name: Adam, params: {} }" \
  --training_steps  "25000,25000" --skip-infer false \
  --learning_rate "0.01,0.001" --use-gpu true --feature_scaling 'cmvn'


# densenet
for l in 8;do
	for k in 16;do
		hparams="add_first_batch_norm=True,freeze_first_batch_norm=False,inital_filters=16,dense_blocks=3,num_layers=${l},growth_rate=${k},add_bottleneck_layer=False,theta=1"
		suffix="_F16_DB3L${l}K${k}_CMN"
		echo '========= $suffix ========='
		bash run.sh --mode train --model densenet \
			--data_dir $data_dir --test_dir ${test_dir} \
			--hparams "$hparams" \
			--suffix "$suffix" \
			--opt "{ name: Adam, params: {} }" \
			--training_steps  "10000,10000,10000,10000" --skip-infer false \
			--learning_rate "0.01,0.001,0.0005,0.0001" --use-gpu true --feature_scaling 'cmn' || exit 1
	done
done
