import argparse
import os
import re
import random
import sys
from collections import defaultdict
import spacy
import itertools
import crosslingual_coreference
import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from nltk.corpus import wordnet as wn
from coref_score import sentence_level_eval_score
from nltk.corpus import words
from utils import readCoNLL12
from checkDepth import getDepth
from baselines.PatInv import generate as PatInvGenerate
from baselines.CAT import generate as CATGenerate
from baselines.SIT import generate as SITGenerate


sys.path.append('./')
sys.path.append('../')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# fix random seed for reproduction
random.seed(10)

stopWords = set(stopwords.words('english'))

tokenizer = TreebankWordTokenizer()
detokenizer = TreebankWordDetokenizer()

from stanfordcorenlp import StanfordCoreNLP

try:
    # TODO: PLEASE FILL IN THE MODEL_DIR BEFORE RUNNING THE PROGRAM!!!
    MODEL_DIR = r'/DIR-TO-SOMETHING-LIKE-THIS/stanford-corenlp-4.4.0'
    coreNLP = StanfordCoreNLP('http://localhost', port=9001, lang="en")
except:
    raise ConnectionRefusedError("Please make sure you have download stanford-corenlp-4.4.0+ "
                                 "and launch a server from there.\n"
                                 "The port is set to be 9001. Please make sure there are no ports conflict.")

checkDepth = False


def setup_nlp_pipeline():
    try:
        # install the model by running `python -m spacy download en_core_web_sm`
        nlp = spacy.load("en_core_web_sm")
    except:
        raise FileNotFoundError("Cannot find spacy model. Run `python -m spacy download en_core_web_sm` to download.")

    nlp.add_pipe(
        "xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2, "device": -1}
    )
    return nlp


def get_syns_dict():
    SIM_DICT_FILE = "similarity_dict.txt"

    sim_dict = {}
    with open(SIM_DICT_FILE, 'r') as f:
        lines = f.readlines()
        for l in lines:
            sim_dict[l.split()[0]] = l.split()[1:]
    return sim_dict


def get_coreference(text, pipeline=None):
    if pipeline is None:
        pipeline = setup_nlp_pipeline()

    doc = pipeline(text)
    coref_text = []
    coref_index = []

    i = 0
    for cluster in doc._.coref_clusters:
        temp_words = []
        temp_index = []
        # print(f"Cluster {i}")
        for item in cluster:
            start, end = item
            temp_words.append(doc[start:end + 1].text)
            temp_index.append('-'.join([str(start), str(end)]))
            # print(doc[start:end + 1])
        # print()
        i += 1
        coref_index.append(' == '.join(temp_index))
        coref_text.append(' == '.join(temp_words))
        # coref_set_plain_text.add(temp_words)
    return coref_index, coref_text


def getStandardCoref(doc):
    # assert pipeline is not None
    # doc = pipeline(text)

    return doc._.coref_clusters


def printCorefText(doc, coref):
    coref_text = []
    for cid, cluster in enumerate(coref, 1):
        cur_cluster = []
        for item in cluster:
            start, end = item
            try:
                cur_cluster.append(doc[start:end + 1].text)
            except:
                cur_cluster.append(' '.join(doc[start:end + 1]))
        print(f"- Cluster {cid}: {', '.join(cur_cluster)}")
        coref_text.append(cur_cluster)
    return coref_text


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return None


def get_syn_ant(word, tag):
    wn_pos = get_wordnet_pos(tag)
    synset = wn.synsets(word, pos=wn_pos)

    synonyms = []
    antonyms = []

    for syn in synset:
        # print(syn, syn.definition())
        for l in syn.lemmas():
            # wn.synset(l.name())
            # print("similarity: ", syn.lch_similarity())
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    synonyms = set(filter(lambda x: len(re.split(r'[-_]', x)) == 1, synonyms))
    antonyms = set(filter(lambda x: len(re.split(r'[-_]', x)) == 1, antonyms))

    synonyms -= {word}
    antonyms -= {word}

    # print("\nSynonyms of {}: [{}]".format(word, ','.join(synonyms)))
    # print("\nAntonyms of {}: [{}]".format(word, ','.join(antonyms)))

    return list(synonyms), list(antonyms)


def getCorefRelatedIndex(sent, coref):
    def getCorefIndex(coref_list):
        corefIndex = set()
        for cluster in coref_list:
            # print('cluster', cluster)
            for item in cluster:
                # print('item', item)
                corefIndex |= set([x for x in range(item[0], item[1] + 1)])
        return corefIndex

    depTree = coreNLP.dependency_parse(sent)
    corefIndex = getCorefIndex(coref)

    FixedIndex = set()
    for item in depTree[1:]:
        relation, fromID, toID = item
        if toID in corefIndex:
            FixedIndex.add(fromID)

    FixedIndex |= corefIndex

    return FixedIndex


def generate_WordReplace_Baseline(oriTokens, coref, max_num=5):
    sim_dict = get_syns_dict()
    pos_inf = nltk.tag.pos_tag(oriTokens)

    new_sentences, new_tokens, masked_indexes = [], [], []
    replaced = []
    for idx, (word, tag) in enumerate(pos_inf):
        if word in sim_dict:
            masked_indexes.append((idx, tag))

    num_new_sent = 0
    for (masked_index, tag) in masked_indexes:
        if num_new_sent >= max_num:
            break

        original_word = oriTokens[masked_index]

        # only replace noun, adjective, number
        if tag.startswith('NN') or tag.startswith('JJ') or tag == 'CD' or tag.startswith('VV'):

            # generate similar sentences
            for similar_word in sim_dict[original_word]:
                # Directly replace without check tag's consistency
                oriTokens[masked_index] = similar_word
                new_sentence = detokenizer.detokenize(oriTokens)
                new_sentence = new_sentence.replace(' /', '')
                new_sentences.append(new_sentence)
                new_tokens.append(oriTokens)
                replaced.append((original_word, similar_word))
                num_new_sent += 1

            oriTokens[masked_index] = original_word

    return new_sentences, new_tokens, replaced


def generate_WordReplace_FilterTag(oriTokens, coref, max_num=5):
    sim_dict = get_syns_dict()

    pos_inf = nltk.tag.pos_tag(oriTokens)

    new_sentences, new_tokens, masked_indexes = [], [], []
    replaced = []
    for idx, (word, tag) in enumerate(pos_inf):
        if word in sim_dict:
            masked_indexes.append((idx, tag))

    num_new_sent = 0
    for (masked_index, tag) in masked_indexes:
        if num_new_sent >= max_num:
            break

        original_word = oriTokens[masked_index]

        # only replace noun, adjective, number
        if tag.startswith('NN') or tag.startswith('JJ') or tag == 'CD' or tag.startswith('VV'):

            # generate similar sentences
            for similar_word in sim_dict[original_word]:
                oriTokens[masked_index] = similar_word
                new_pos_inf = nltk.tag.pos_tag(oriTokens)

                # check that tag is still same type, including the singular or not
                if new_pos_inf[masked_index][1] == tag:
                    new_sentence = detokenizer.detokenize(oriTokens)
                    new_sentence = new_sentence.replace(' /', '')
                    new_sentences.append(new_sentence)
                    new_tokens.append(oriTokens)
                    replaced.append((original_word, similar_word))
                    num_new_sent += 1

            oriTokens[masked_index] = original_word

    return new_sentences, new_tokens, replaced


def generate_WordReplace_SynAnt(oriTokens, coref, max_num=5):
    pos_inf = nltk.tag.pos_tag(oriTokens)

    new_sentences, new_tokens, masked_indexes = [], [], []
    replaced = []
    candidates = defaultdict(set)

    for idx, (word, tag) in enumerate(pos_inf):
        syns, ants = get_syn_ant(word, tag)
        if len(syns) > 0 or len(ants) > 0:
            masked_indexes.append((idx, tag))
            # at most use max_num synonyms and max_num antonyms
            candidates[word] = set(syns[:max_num]) | set(ants[:max_num])

    num_new_sent = 0
    for (masked_index, tag) in masked_indexes:
        if num_new_sent >= max_num:
            break

        original_word = oriTokens[masked_index]

        # only replace noun, adjective, number
        if tag.startswith('NN') or tag.startswith('JJ') or tag == 'CD' or tag.startswith('VV'):
            for similar_word in candidates[original_word]:
                oriTokens[masked_index] = similar_word
                new_pos_inf = nltk.tag.pos_tag(oriTokens)

                # check that tag is still same type, including the singular or not
                if new_pos_inf[masked_index][1] == tag:
                    print("similar_word", similar_word)
                    new_sentence = detokenizer.detokenize(oriTokens)
                    new_sentence = new_sentence.replace(' /', '')
                    new_sentences.append(new_sentence)
                    new_tokens.append(oriTokens)
                    replaced.append((original_word, similar_word))
                    num_new_sent += 1

        oriTokens[masked_index] = original_word

    return new_sentences, new_tokens, replaced


def generate_SIT(oriTokens, coref, max_num=5):
    new_sentences, new_tokens, replaced = SITGenerate(detokenizer.detokenize(oriTokens))
    return new_sentences, new_tokens, replaced


def generate_CAT(oriTokens, coref, max_num=5):
    new_sentences, new_tokens, replaced = CATGenerate(detokenizer.detokenize(oriTokens))
    return new_sentences, new_tokens, replaced


def generate_PatInv(oriTokens, coref, max_num=5):
    new_sentences, new_tokens, replaced = PatInvGenerate(detokenizer.detokenize(oriTokens))
    return new_sentences, new_tokens, replaced


def generate_Crest(oriTokens, coref, max_num=5):
    pos_inf = nltk.tag.pos_tag(oriTokens)

    new_sentences, new_tokens, masked_indexes = [], [], []
    replaced = []
    candidates = defaultdict(set)

    # Find coref dependent index
    FixedIndex = getCorefRelatedIndex(detokenizer.detokenize(oriTokens), coref)

    for idx, (word, tag) in enumerate(pos_inf):
        # Do not replace the coref-dependent index
        if idx in FixedIndex:
            continue

        # Skip stopwords
        if word in stopWords:
            continue

        syns, ants = get_syn_ant(word, tag)
        if len(syns) > 0 or len(ants) > 0:
            masked_indexes.append((idx, tag))
            # at most use max_num synonyms and max_num antonyms
            candidates[word] = set(syns[:max_num]) | set(ants[:max_num])

    num_new_sent = 0
    for (masked_index, tag) in masked_indexes:

        original_word = oriTokens[masked_index]
        # print("original_word", original_word)

        # only replace noun, adjective, cardinal number, adverb, and verb
        if tag.startswith('NN') or tag.startswith('JJ') or tag == 'CD' or tag.startswith('RB') or tag.startswith('VB'):
            for similar_word in candidates[original_word]:
                oriTokens[masked_index] = similar_word
                new_pos_inf = nltk.tag.pos_tag(oriTokens)

                # check that tag is still same type, including the singular or not
                if new_pos_inf[masked_index][1] == tag:
                    new_sentence = detokenizer.detokenize(oriTokens)
                    new_sentence = new_sentence.replace(' /', '')
                    new_sentences.append(new_sentence)
                    new_tokens.append(oriTokens)
                    replaced.append((original_word, similar_word))
                    num_new_sent += 1

        oriTokens[masked_index] = original_word

    return new_sentences, new_tokens, replaced


def calMUC(predicted_clusters, gold_clusters):
    """
    the link based MUC

    Parameters
    ------
        predicted_clusters      list(list)       predicted clusters
        gold_clusters           list(list)       gold clusters
    Return
    ------
        tuple(float)    precision, recall, F1, TP,
    """

    def processCluster(coref):
        newCoref = []
        for cluster in coref:
            cur_cluster = []
            for item in cluster:
                cur_cluster.append('-'.join([str(x) for x in item]))
            newCoref.append(cur_cluster)
        return newCoref

    predicted_clusters = processCluster(predicted_clusters)
    gold_clusters = processCluster(gold_clusters)

    pred_edges = set()
    for cluster in predicted_clusters:
        pred_edges |= set(itertools.combinations(sorted(cluster), 2))
    gold_edges = set()
    for cluster in gold_clusters:
        gold_edges |= set(itertools.combinations(sorted(cluster), 2))
    correct_edges = gold_edges & pred_edges

    precision = len(correct_edges) / len(pred_edges) if len(pred_edges) != 0 else 0.0
    recall = len(correct_edges) / len(gold_edges) if len(gold_edges) != 0 else 0.0
    f1 = getF1(precision, recall)

    return precision, recall, f1


def getF1(precision, recall):
    return precision * recall * 2 / (precision + recall) if precision + recall != 0 else 0.0


def processCluster(coref):
    newCoref = []
    for cluster in coref:
        cur_cluster = []
        for item in cluster:
            cur_cluster.append('-'.join([str(x) for x in item]))
        newCoref.append(cur_cluster)
    return newCoref


def evaluate(docID, oriToken, oriCluster, newToken, newCluster, METRIC="blanc", extra_label=None):
    def cluster2json(docID, tokens, cluster):
        return {"doc_key": docID, "sentences": [tokens], "clusters": cluster}

    oriJson = cluster2json(docID, oriToken, oriCluster)
    newJson = cluster2json(docID, newToken, newCluster)

    oriFilename = "temp_ori.json" if extra_label is None else f"temp_ori_{extra_label}.json"
    newFilename = "temp_new.json" if extra_label is None else f"temp_new_{extra_label}.json"

    conll_dir = "../Output/conll"
    os.makedirs(conll_dir, exist_ok=True)
    all_predict_file = os.path.join(conll_dir, "all_predict.conll") if extra_label is None \
        else os.path.join(conll_dir, f"all_predict_{extra_label}.conll")
    all_gold_file = os.path.join(conll_dir, "all_gold.conll") if extra_label is None \
        else os.path.join(conll_dir, f"all_gold_{extra_label}.conll")

    with open(oriFilename, 'w') as f:
        json.dump(oriJson, f)
    with open(newFilename, 'w') as f:
        json.dump(newJson, f)

    result = sentence_level_eval_score(oriFilename, newFilename, METRIC=METRIC,
                                       all_predict_file=all_predict_file, all_gold_file=all_gold_file)

    os.remove(oriFilename)
    os.remove(newFilename)
    return result[1], result[0], result[2]


def Coref_testing(nlp, oriSentencePairs, genMethodName='Crest', output_fn='../output.tsv'):
    output_tsv = open(output_fn, 'w')
    if genMethodName == 'Crest':
        output_tsv.write("\t".join(['doc_id', 'OID', 'GID', 'oriSent', 'genSent', 'repToken',
                                    'oriConsistent',
                                    'pairConsistent',
                                    'oriCoref', 'oriCorefText',
                                    'newCoref', 'newCorefText',
                                    'oriPrecision', 'oriRecall',
                                    'pairPrecision', 'pairRecall',
                                    'oriDepth', 'newDepth',
                                    'depthConsistent'
                                    ]) + '\n')
    else:
        output_tsv.write("\t".join(['doc_id', 'OID', 'GID', 'oriSent', 'genSent', 'repToken',
                                    'oriConsistent',
                                    'pairConsistent',
                                    'oriCoref', 'oriCorefText',
                                    'newCoref', 'newCorefText',
                                    'oriPrecision', 'oriRecall',
                                    'pairPrecision', 'pairRecall'
                                    ]) + '\n')

    # initialize stats variables
    num_generated_pairs = 0
    num_origin_fail = 0
    num_fail = 0

    OID = 0
    # For every original sentence
    for doc_id, oriTokens, oriSent, goldCoref in oriSentencePairs:
        # analyze original sentence
        oriDoc = nlp(oriSent)
        oriCoref = getStandardCoref(oriDoc)

        cur_ori_precision, cur_ori_recall, cur_ori_f1 = \
            evaluate(doc_id, oriSent, oriCoref, oriSent, goldCoref, METRIC="blanc", extra_label=genMethodName)

        oriConsistent = (cur_ori_precision == 100.0 and cur_ori_recall == 100.0)
        # print("oriConsistent", oriConsistent)

        # record depth to the closet nested NPs of the corefs
        oriDept = set()
        if genMethodName == 'Crest':
            if checkDepth:
                for cluster in oriCoref:
                    for ent in cluster:
                        # print([x.text for x in oriDoc[ent[0]: ent[1]+1]])
                        oriDept |= getDepth(oriSent, [x.text for x in oriDoc[ent[0]: ent[1] + 1]])

        # generate sentences
        genMethod = getattr(sys.modules[__name__], "generate_{}".format(genMethodName))
        new_sentences, new_tokens, replaced_pairs = genMethod(oriTokens, oriCoref, max_num=10)

        GID = 0
        for newSent, newToken, replaced_pair in zip(new_sentences, new_tokens, replaced_pairs):
            print(f"\n<Pair {num_generated_pairs}>")
            print('Origin  : ', oriSent)
            print('Generate: ', newSent)
            print(f"Replace: {replaced_pair[0]} -> {replaced_pair[1]}")

            print("=" * 20)
            print("oriConsistent", oriConsistent)

            newDoc = nlp(newSent)

            # get coreference of generated sentence
            newCoref = getStandardCoref(newDoc)

            cur_new_precision, cur_new_recall, cur_new_f1 = \
                evaluate(doc_id, newToken, newCoref, oriTokens, oriCoref, METRIC="blanc", extra_label=genMethodName)

            # pairConsistent = (oriCoref == newCoref and len(oriCoref) > 0 and len(newCoref) > 0)
            pairConsistent = (cur_new_precision == 100.0 and cur_new_recall == 100.0)

            # calculate depth
            newDept = set()
            depthConsistent = None

            if genMethodName == 'Crest':
                if checkDepth:
                    for cluster in newCoref:
                        for ent in cluster:
                            newDept |= getDepth(newSent, [x.text for x in newDoc[ent[0]: ent[1] + 1]])
                # check depth
                depthConsistent = newDept == oriDept

            # Start printing information
            print("> Origin sentence's Coref:")
            print(oriCoref)
            oriCoref_text = printCorefText(oriDoc, oriCoref)

            print("\n> Generated sentence's Coref:")
            print(newCoref)
            newCoref_text = printCorefText(newDoc, newCoref)

            # OutputVal bug report
            if not pairConsistent:
                num_fail += 1
                print("[Bug found!] Inconsistent!")
            else:
                print("[Pass]")
            # End of print

            print("Origin  : Precision: {:.2f} | Recall: {:.2f} | F1: {:.2f}".format(cur_ori_precision, cur_ori_recall,
                                                                                     cur_ori_f1))
            print(
                "Generate  : Precision: {:.2f} | Recall: {:.2f} | F1: {:.2f}".format(cur_new_precision, cur_new_recall,
                                                                                     cur_new_f1))
            print("=" * 20)

            num_generated_pairs += 1

            if genMethodName == 'Crest':
                output_tsv.write('\t'.join([doc_id, str(OID), str(GID), oriSent, newSent,
                                            f'{replaced_pair[0]} -> {replaced_pair[1]}',
                                            str(oriConsistent),
                                            str(pairConsistent),
                                            str(oriCoref), str(oriCoref_text),
                                            str(newCoref), str(newCoref_text),
                                            str(cur_ori_precision), str(cur_ori_recall),
                                            str(cur_new_precision), str(cur_new_recall),
                                            str(oriDept), str(newDept),
                                            str(depthConsistent)
                                            ]) + '\n')
            else:
                output_tsv.write('\t'.join([doc_id, str(OID), str(GID), oriSent, newSent,
                                            f'{replaced_pair[0]} -> {replaced_pair[1]}',
                                            str(oriConsistent),
                                            str(pairConsistent),
                                            str(oriCoref), str(oriCoref_text),
                                            str(newCoref), str(newCoref_text),
                                            str(cur_ori_precision), str(cur_ori_recall),
                                            str(cur_new_precision), str(cur_new_recall),
                                            ]) + '\n')

            GID += 1
            # End of iteration of generated sentences

        OID += 1
        # End of iteration for the current original sentence

    print("\n\n")
    print("=" * 20)
    print("Summary: \n" \
          "Number of origin sentence: {} | Failed: {}\n" \
          "Number of generated pairs: {} | Failed: {}".format(
        len(oriSentencePairs), num_origin_fail,
        num_generated_pairs, num_fail
    ))

    output_tsv.close()



if __name__ == '__main__':

    OutputDir = '../Output'
    # read Output
    oriSentencePairs = readCoNLL12()
    # oriSentencePairs = oriSentencePairs[:2]
    oriSentencePairs = random.sample(oriSentencePairs, k=1)

    # load nlp
    nlp = setup_nlp_pipeline()

    # choose generation methods
    genMethodNameList = ['SIT']  # 'PatInv', 'CAT', 'SIT'

    for genMethodName in genMethodNameList:
        # set up output tsv file
        output_tsv_fn = os.path.join(OutputDir, '{}.tsv'.format(genMethodName))
        Coref_testing(nlp, oriSentencePairs, genMethodName=genMethodName, output_fn=output_tsv_fn)
