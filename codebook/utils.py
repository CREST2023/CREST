import json


def readCoNLL12():
    def clean(sent: str):
        sent = sent.strip('\n')
        sent = sent.replace(" '", "'")
        sent = sent.replace(" /.", ".")
        sent = sent.replace(" ,", ',')
        sent = sent.replace(' .', '.')
        sent = sent.replace(" - ", '-')
        sent = sent.replace("`` ", "\"")
        sent = sent.replace("''", "\"")
        sent = sent.replace("`", "\'")
        sent = sent.replace(" n't", "n't")
        sent = sent.replace(" ?", '?')
        sent = sent.replace(" ;", ',')
        sent = sent.replace(" & ", '&')
        sent = sent.replace("\"", "'")
        return sent

    def isValid(sent: str):
        # Here we remove the sentence with symbols to avoid the inconsistency caused by tokenization
        invalid_symb_set = {"-", "#", '%', '&', '*', "'", ':', "\""}
        for sym in invalid_symb_set:
            if sym in sent:
                return False
        return True

    sentencePairs = []
    num_lines = 0
    num_invalid = 0

    news_datasets = {'bn', 'mz', 'nw'}
    # | bn | Broadcast News  |
    # | mz | Magazine (Newswire) |
    # | nw | Newswire  |

    with open('../rawCoNLL12/test.english.v4_gold_conll.sen.json', 'r') as conll12:
        for line in conll12.readlines():
            num_lines += 1
            sent_dict = json.loads(line)
            docID = sent_dict['doc_id']
            token = sent_dict['sentence']
            sent = " ".join(token)
            if not isValid(sent) or docID[:2] not in news_datasets or not sent.endswith('.'):
                num_invalid += 1
                continue

            sent = clean(sent)
            gold_cluster = sent_dict['cluster']

            sentencePairs.append((docID, token, sent, gold_cluster))
    print(
        f"Read {num_lines} lines from conll12/dev.english.v4_gold_conll.sen.json.")

    return sentencePairs
