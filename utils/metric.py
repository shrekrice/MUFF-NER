# -*- coding: utf-8 -*-

import sys


def get_ner_fmeasure(golden_lists, predict_lists, label_type="", printnum=True):
    sent_num = len(golden_lists)
    golden_full, predict_full, right_full, right_tag, all_tag = [], [], [], 0, 0

    for idx in range(sent_num):
        golden_list, predict_list = golden_lists[idx], predict_lists[idx]
        right_tag += sum(g == p for g, p in zip(golden_list, predict_list))
        all_tag += len(golden_list)

        gold_matrix = get_ner_matrix(golden_list, label_type)
        pred_matrix = get_ner_matrix(predict_list, label_type)

        right_full += list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix

    right_num, golden_num, predict_num = len(right_full), len(golden_full), len(predict_full)
    precision = right_num / predict_num if predict_num > 0 else -1
    recall = right_num / golden_num if golden_num > 0 else -1
    f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else -1
    accuracy = right_tag / all_tag

    if printnum:
        print(f"gold_num = {golden_num}, pred_num = {predict_num}, right_num = {right_num}")

    return accuracy, precision, recall, f_measure


def get_ner_matrix(label_list, label_type):
    if label_type == "":
        return get_ner_BMES(label_list)
    else:
        return get_ner_BIO(label_list)


def get_ner_BMES(label_list):
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper() if label_list[i] else []
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
            index_tag = current_label.replace(begin_label, "", 1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(single_label, "", 1) + '[' + str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix
def read_sentence(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    sentences, labels, sentence, label = [], [], [], []

    for line in lines:
        if len(line) < 2:
            sentences.append(sentence)
            labels.append(label)
            sentence, label = [], []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            label.append(pair[-1])

    return sentences, labels
def read_two_label_sentence(input_file, pred_col=-1):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    sentences, golden_labels, predict_labels, sentence, golden_label, predict_label = [], [], [], [], [], []

    for line in lines:
        if "#score#" in line:
            continue
        if len(line) < 2:
            sentences.append(sentence)
            golden_labels.append(golden_label)
            predict_labels.append(predict_label)
            sentence, golden_label, predict_label = [], [], []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            golden_label.append(pair[1])
            predict_label.append(pair[pred_col])

    return sentences, golden_labels, predict_labels


def f_measure_from_file(golden_file, predict_file, label_type=""):
    print(f"Get f measure from file: {golden_file}, {predict_file}")
    print(f"Label format: {label_type}")
    golden_sent, golden_labels = read_sentence(golden_file)
    predict_sent, predict_labels = read_sentence(predict_file)
    acc, P, R, F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    print(f"Acc: {acc}, P: {P}, R: {R}, F: {F}")


def f_measure_from_single_file(two_label_file, label_type="", pred_col=-1):
    sent, golden_labels, predict_labels = read_two_label_sentence(two_label_file, pred_col)
    P, R, F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    print(f"P: {P}, R: {R}, F: {F}")


if __name__ == '__main__':
    if len(sys.argv) == 3:
        f_measure_from_single_file(sys.argv[1], "", int(sys.argv[2]))
    else:
        f_measure_from_single_file(sys.argv[1], "")
