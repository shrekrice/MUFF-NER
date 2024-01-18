import time
import sys
import argparse
import random
import torch
import pickle
import os
from torch import optim
from model import SeqModel, DecodeModel
from utils import set_seed, lr_decay, batchify_with_label, predict_check, evaluate, data_initialization
from data import Data


def train(data, save_model_dir, seg=True):
    print(f"Training with {data.model_type} model.")

    model = SeqModel(data)
    print("building model.")

    optimizer = optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=data.HP_lr)

    best_dev, best_dev_p, best_dev_r = -1, -1, -1
    best_test, best_test_p, best_test_r = -1, -1, -1

    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        print(f"Epoch: {idx + 1}/{data.HP_iteration}")
        optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)

        instance_count, sample_loss, batch_loss, total_loss, right_token, whole_token = 0, 0, 0, 0, 0, 0
        random.shuffle(data.train_Ids)

        model.train()
        model.zero_grad()

        for batch_id in range(0, len(data.train_Ids), data.HP_batch_size):
            start, end = batch_id, min(batch_id + data.HP_batch_size, len(data.train_Ids))
            instance = data.train_Ids[start:end]
            words = data.train_texts[start:end]

            if not instance:
                continue


    print(f"Best dev score: p:{best_dev_p}, r:{best_dev_r}, f:{best_dev}")
    print(f"Test score: p:{best_test_p}, r:{best_test_r}, f:{best_test}")
    gc.collect()

    with open(data.result_file, "a") as f:
        f.write(f"{save_model_dir}\n")
        f.write(f"Best dev score: p:{best_dev_p}, r:{best_dev_r}, f:{best_dev}\n")
        f.write(f"Test score: p:{best_test_p}, r:{best_test_r}, f:{best_test}\n\n")


def load_model_decode(model_dir, data, name, gpu, seg=True):
    data.HP_gpu = gpu
    print(f"Load Model from file: {model_dir}")
    model = DecodeModel(data)
    model.load_state_dict(torch.load(model_dir))

    print(f"Decode {name} data ...")
    start_time = time.time()
    speed, acc, p, r, f, pred_results, gazs = evaluate(data, model, name)
    end_time = time.time()
    time_cost = end_time - start_time

    if seg:
        print(f"{name}: time:{time_cost:.2f}s, speed:{speed:.2f}st/s; acc: {acc:.4f}, p: {p:.4f}, r: {r:.4f}, f: {f:.4f}")
    else:
        print(f"{name}: time:{time_cost:.2f}s, speed:{speed:.2f}st/s; acc: {acc:.4f}")

    return pred_results


def print_results(pred, modelname=""):
    toprint = [" ".join(sen) + '\n' for sen in pred]
    with open(f"{modelname}_labels.txt", 'w') as f:
        f.writelines(toprint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', default='None')
    parser.add_argument('--status', choices=['train', 'test'], default='train')
    parser.add_argument('--modelpath', default="save_model/")
    parser.add_argument('--modelname', default="model")
    parser.add_argument('--train', default="ResumeNER/train.char.bmes")
    parser.add_argument('--dev', default="ResumeNER/dev.char.bmes")
    parser.add_argument('--test', default="ResumeNER/test.char.bmes")
    parser.add_argument('--seg', default="True")
    parser.add_argument('--output')
    parser.add_argument('--seed', default=1023, type=int)
    parser.add_argument('--labelcomment', default="")
    parser.add_argument('--resultfile', default="result/result.txt")
    parser.add_argument('--num_iter', default=100, type=int)
    parser.add_argument('--num_layer', default=4, type=int)
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--model_type', default='lstm')
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--use_biword', dest='use_biword', action='store_true', default=False)
    parser.add_argument('--use_char', dest='use_char', action='store_true', default=False)
    parser.add_argument('--use_count',default=True)
    parser.add_argument('--use_bert', default=True)

    args = parser.parse_args()
    seed_num = args.seed
    set_seed(seed_num)

    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    output_file = args.output
    if args.seg.lower() == "true":
        seg = True
    else:
        seg = False
    status = args.status.lower()

    save_model_dir = f"{args.modelpath}{args.modelname}"
    save_data_name = args.savedset
    gpu = torch.cuda.is_available()

    char_emb = "../../gigaword_chn.all.a2b.uni.ite50.vec"
    bichar_emb = "../../gigaword_chn.all.a2b.bi.ite50.vec"
    gaz_file = "../../ctb.50d.vec"

    sys.stdout.flush()

    if status == 'train':
        if os.path.exists(save_data_name):
            print('Loading processed data')
            with open(save_data_name, 'rb') as fp:
                data = pickle.load(fp)
            data.HP_num_layer = args.num_layer
            data.HP_batch_size = args.batch_size
            data.HP_iteration = args.num_iter
            data.label_comment = args.labelcomment
            data.result_file = args.resultfile
            data.HP_lr = args.lr
            data.use_bigram = args.use_biword
            data.HP_use_char = args.use_char
            data.HP_hidden_dim = args.hidden_dim
            data.HP_dropout = args.drop
            data.HP_use_count = args.use_count
            data.model_type = args.model_type
            data.use_bert = args.use_bert
        else:
            data = Data()
            data.HP_gpu = gpu
            data.HP_use_char = args.use_char
            data.HP_batch_size = args.batch_size
            data.HP_num_layer = args.num_layer
            data.HP_iteration = args.num_iter
            data.use_bigram = args.use_biword
            data.HP_dropout = args.drop
            data.norm_gaz_emb = False
            data.HP_fix_gaz_emb = False
            data.label_comment = args.labelcomment
            data.result_file = args.resultfile
            data.HP_lr = args.lr
            data.HP_hidden_dim = args.hidden_dim
            data.HP_use_count = args.use_count
            data.model_type = args.model_type
            data.use_bert = args.use_bert
            data_initialization(data, gaz_file, train_file, dev_file, test_file)
            data.generate_instance_with_gaz(train_file, 'train')
            data.generate_instance_with_gaz(dev_file, 'dev')
            data.generate_instance_with_gaz(test_file, 'test')
            data.build_word_pretrain_emb(char_emb)
            data.build_biword_pretrain_emb(bichar_emb)
            data.build_gaz_pretrain_emb(gaz_file)

            print('Dumping data')
            with open(save_data_name, 'wb') as f:
                pickle.dump(data, f)
            set_seed(seed_num)
        print(f'data.use_biword={data.use_bigram}')
        train(data, save_model_dir, seg)
    elif status == 'test':
        print('Loading processed data')
        with open(save_data_name, 'rb') as fp:
            data = pickle.load(fp)
        data.HP_num_layer = args.num_layer
        data.HP_iteration = args.num_iter
        data.label_comment = args.labelcomment
        data.result_file = args.resultfile
        data.HP_lr = args.lr
        data.use_bigram = args.use_biword
        data.HP_use_char = args.use_char
        data.model_type = args.model_type
        data.HP_hidden_dim = args.hidden_dim
        data.HP_use_count = args.use_count
        data.generate_instance_with_gaz(test_file, 'test')
        load_model_decode(save_model_dir, data, 'test', gpu, seg)
    else:
        print("Invalid argument!")
