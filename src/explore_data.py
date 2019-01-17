import os
import json
import numpy as np

from data_handler import DataLoader

load_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__))
                                    + '/../dat/fast_load')
loader = DataLoader(load_dir, tst_size=0)
wrong_beg_end = []
wrong_cont = []
for j, i in enumerate(loader.ids):
    feat_seq, align_seq, _ = loader.load(i)
    try:
        assert align_seq[0][0] == 0
        assert align_seq[-1][0] + align_seq[-1][1] == feat_seq.shape[0]
    except AssertionError:
        wrong_beg_end.append(i)
        print('Wrong beg/end:', j, i)
    try:
        for k, token in enumerate(align_seq[1:]):
            assert align_seq[k][0] + align_seq[k][1] == token[0]
    except AssertionError:
        wrong_cont.append(i)
        print('Wrong cont.:  ', j, i)
output_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__))
                                      + '/../dat/corrupted')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(output_dir + '/wrong_beg_end.json', 'w') as handle:
    json.dump(wrong_beg_end, handle, indent=4)
with open(output_dir + '/wrong_cont.json', 'w') as handle:
    json.dump(wrong_cont, handle, indent=4)
ids_cleaned = sorted(list(set(loader.ids) - set(wrong_beg_end) - set(wrong_cont)))
with open(load_dir + '/utterances.json', 'w') as handle:
    json.dump(ids_cleaned, handle, indent=4)
