
. ./path.sh

set -e

data_dir=~/Data/TF_Speech/speech_commands_extend
test_dir=~/Data/TF_Speech/test_extend

# cp -r ~/Data/TF_Speech/speech_commands $data_dir
# cp -r ~/Data/TF_Speech/test $test_dir

datadir=data
mkdir -p $datadir

# find $data_dir -iname "*.wav" | grep -v background | grep -v "_sp" | grep -v "_ps" | grep -v "_vp" >$datadir/train.wavlist
# find $test_dir -iname "*.wav" | grep -v "_sp" | grep -v "_ps" | grep -v "_vp" >$datadir/test.wavlist

# for x in train test;do
for x in test;do
    mkdir -p $datadir/$x
    python -c "
import sys
import os

assert len(sys.argv) == 3

wavlist, data_dir = sys.argv[1:]

with open(data_dir + '/wav.scp', 'w') as wf:
  with open(data_dir + '/text', 'w') as tf:
    with open(data_dir + '/utt2spk', 'w') as uf:
        with open(wavlist) as f:
          for line in f:
            wav_file = line.strip()
            feilds = wav_file.split('/')
            user_id = feilds[-1].split('_')[0]
            key = user_id + '_' + feilds[-2] + '_' + feilds[-1].replace('.wav', '')
            print >>wf, '%s %s' % (key, wav_file)
            print >>tf, '%s test' % (key)
            print >>uf, '%s %s' % (key, user_id)
    " $datadir/$x.wavlist $datadir/$x || exit 1

    sort $datadir/$x/utt2spk -o $datadir/$x/utt2spk
    utils/utt2spk_to_spk2utt.pl $datadir/$x/utt2spk >$datadir/$x/spk2utt
    # utils/fix_data_dir.sh $datadir/$x

    if [ ! -f $datadir/$x/silence_probs.ark ];then
        # steps/make_mfcc.sh --nj 8 --mfcc-config endpointer/mfcc_hires.conf $datadir/$x
        # steps/compute_cmvn_stats.sh $datadir/$x
        steps/nnet3/compute_output.sh --apply_exp true $datadir/$x endpointer $datadir/$x/output
        copy-feats scp:$datadir/$x/output/output.scp ark,t:$datadir/$x/silence_probs.ark
    fi
done

# ./bin/data_argument.py data/train/silence_probs.ark data/train.wavlist
# utils/run.pl JOB=1:6 data/train/log/data_argument.JOB.log \
#     ./bin/data_argument.py data/train/silence_probs.ark data/train.wavlist_JOB
# ./bin/data_argument.py data/test/silence_probs.ark data/test.wavlist

# ./utils/split_scp.pl data/test.wavlist data/test.wavlist_1 data/test.wavlist_2 data/test.wavlist_3 data/test.wavlist_4 data/test.wavlist_5 data/test.wavlist_6
# utils/run.pl JOB=1:6 data/test/log/data_argument.JOB.log ./bin/data_argument.py data/test/silence_probs.ark data/test.wavlist_JOB

echo "$0: DONE"
