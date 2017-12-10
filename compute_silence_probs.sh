
. ./path.sh

datadir=test/data
mkdir -p $datadir


find `pwd`/test/sample  -iname "*.wav" >test/test.wavlist.100
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
            key = feilds[-2] + '_' + feilds[-1].replace('.wav', '')
            print >>wf, '%s %s' % (key, wav_file)
            print >>tf, '%s test' % (key)
            print >>uf, '%s %s' % (key, user_id)
" test/test.wavlist.100 $datadir || exit 1

sort $datadir/utt2spk -o $datadir/utt2spk
utils/utt2spk_to_spk2utt.pl $datadir/utt2spk >$datadir/spk2utt
utils/fix_data_dir.sh $datadir

steps/make_mfcc.sh --nj 8 --mfcc-config endpointer/mfcc_hires.conf $datadir
steps/compute_cmvn_stats.sh $datadir
# nnet3-compute --apply-exp=true endpointer/final.raw scp:$datadir/feats.scp ark,t:$datadir/silence_probs.ark
steps/nnet3/compute_output.sh --apply_exp true $datadir endpointer $datadir/output

copy-feats scp:$datadir/output/output.scp ark,t:$datadir/silence_probs2.ark

