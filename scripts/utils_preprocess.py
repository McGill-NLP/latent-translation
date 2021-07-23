import csv
import os
from collections import defaultdict
import argparse
import json

def pawsx_preprocess(args):
    def _preprocess_one_file(infile, outfile, remove_label=False):
        data = []
        for i, line in enumerate(open(infile, 'r')):
            if i == 0:
                continue
            items = line.strip().split('\t')
            sent1 = ' '.join(items[1].strip().split(' '))
            sent2 = ' '.join(items[2].strip().split(' '))
            label = items[3]
            data.append([sent1, sent2, label])

        with open(outfile, 'w') as fout:
            writer = csv.writer(fout, delimiter='\t')
            for sent1, sent2, label in data:
                if remove_label:
                    writer.writerow([sent1, sent2])
                else:
                    writer.writerow([sent1, sent2, label])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    split2file = {'train': 'train', 'test': 'test_2k', 'dev': 'dev_2k'}
    for lang in ['en', 'de', 'es', 'fr', 'ja', 'ko', 'zh']:
        for split in ['train', 'test', 'dev']:
            if split == 'train' and lang != 'en':
                continue
            file = split2file[split]
            infile = os.path.join(args.data_dir, lang, "{}.tsv".format(file))
            outfile = os.path.join(args.output_dir, "{}-{}.tsv".format(split, lang))
            _preprocess_one_file(infile, outfile, remove_label=(split == 'test'))
            print(f'finish preprocessing {outfile}')


def xnli_preprocess(args):
    def _preprocess_file(infile, output_dir, split):
        all_langs = defaultdict(list)
        for i, line in enumerate(open(infile, 'r')):
            if i == 0:
                continue

            items = line.strip().split('\t')
            lang = items[0].strip()
            label = "contradiction" if items[1].strip() == "contradictory" else items[1].strip()
            sent1 = ' '.join(items[6].strip().split(' '))
            sent2 = ' '.join(items[7].strip().split(' '))
            all_langs[lang].append((sent1, sent2, label))
        print(f'# langs={len(all_langs)}')
        for lang, pairs in all_langs.items():
            outfile = os.path.join(output_dir, '{}-{}.tsv'.format(split, lang))
            with open(outfile, 'w') as fout:
                writer = csv.writer(fout, delimiter='\t')
                for (sent1, sent2, label) in pairs:
                    if split == 'test':
                        writer.writerow([sent1, sent2])
                    else:
                        writer.writerow([sent1, sent2, label])
            print(f'finish preprocess {outfile}')

    def _preprocess_train_file(infile, outfile):
        with open(outfile, 'w') as fout:
            writer = csv.writer(fout, delimiter='\t')
            for i, line in enumerate(open(infile, 'r')):
                if i == 0:
                    continue

                items = line.strip().split('\t')
                sent1 = ' '.join(items[0].strip().split(' '))
                sent2 = ' '.join(items[1].strip().split(' '))
                label = "contradiction" if items[2].strip() == "contradictory" else items[2].strip()
                writer.writerow([sent1, sent2, label])
        print(f'finish preprocess {outfile}')

    infile = os.path.join(args.data_dir, 'XNLI-MT-1.0/multinli/multinli.train.en.tsv')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    outfile = os.path.join(args.output_dir, 'train-en.tsv')
    _preprocess_train_file(infile, outfile)

    for split in ['test', 'dev']:
        infile = os.path.join(args.data_dir, 'XNLI-1.0/xnli.{}.tsv'.format(split))
        print(f'reading file {infile}')
        _preprocess_file(infile, args.output_dir, split)

def xcopa_preprocess(args):
    letter2int = {"A": 0, "B": 1, "C": 2}
    infile = os.path.join(args.data_dir, 'socialIQa_v1.4_trn.jsonl')
    outfile = os.path.join(args.output_dir, 'train.en.jsonl')
    with open(outfile, 'w') as fout:
        for line in open(infile, 'r'):
            example = json.loads(line)
            new_example = {
                "premise": example["context"],
                "question": example["question"],
                "choice1": example["answerA"],
                "choice2": example["answerB"],
                "choice3": example["answerC"],
                "label": letter2int[example["correct"]]
            }
            fout.write(json.dumps(new_example) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                                            help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                                            help="The output data dir where any processed files will be written to.")
    parser.add_argument("--task", default="panx", type=str, required=True,
                                            help="The task name")
    args = parser.parse_args()

    if args.task == 'pawsx':
        pawsx_preprocess(args)
    elif args.task == 'xnli':
        xnli_preprocess(args)
    elif args.task == 'xcopa':
        xcopa_preprocess(args)
