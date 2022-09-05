import os
import re
import subprocess

METRIC_SET = {
    "muc",
    "bcub",
    "ceafm",
    "ceafe",
    "blanc"
}


def process_command_output(st):
    pattern = "\S\s(.*?)%"
    results = re.findall(pattern, st)
    results = list(map(lambda x: float(x.split(' ')[-1]), results))
    # recall, precision, F1
    return results


def sentence_level_eval_score(gold='./gold.json', predict='./predict.json', METRIC='blanc',
                              all_predict_file='./temp_all_predict.conll', all_gold_file='./temp_all_gold.conll'):
    assert METRIC in METRIC_SET

    output_gold = "{}.conll".format(gold)
    output_predict = "{}.conll".format(predict)

    os.system("python ../corefconversion-master/jsonlines2conll.py -g {} -o {}".format(predict, output_predict))
    os.system("python ../corefconversion-master/jsonlines2conll.py -g {} -o {}".format(gold, output_gold))

    output = subprocess.check_output(
        "perl ../reference-coreference-scorers-master/scorer.pl {} {} {}".format(METRIC, output_predict, output_gold),
        shell=True,
        text=True)

    os.system("cat {} >> {}".format(output_predict, all_predict_file))
    os.system("cat {} >> {}".format(output_gold, all_gold_file))

    # print(output)
    output = output.split('\n')[-3]

    result = process_command_output(output)
    # print(result)
    # print("Recall = {:.2f} | Precision: {:.2f} | F1: {:.2f}".format(result[0], result[1], result[2]))

    os.remove(output_gold)
    os.remove(output_predict)
    return result
