import pathlib
import sys
import os
import re

pattern = re.compile(', ([06]|167|168|169|170|171|172|173|174|175|176|177|178|179|180|181|182|183)\]')

directory = os.fsencode('./dat/fast_load/utterances/')

num_files = 0
found_files = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".json"):
        num_files += 1
        with open(os.path.join(directory, file), 'r') as utter:
            for line in utter:
                result = pattern.search(line)
                if result:
                    found_files += 1
                    print('found %s in %s' % (result, filename))

print("Found unkown phones in %s / %s utterances!" % (found_files, num_files))
