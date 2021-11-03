import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import numpy as np
import torch.utils.data
import torchvision.transforms as transform
import torchvision.datasets as datasets
import argparse
from tqdm import tqdm
import time
import dill as pickle
import math
import random
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TranslationDataset
import sys
sys.path.append("..")
from models.students.transformer import ArchNetTransformer as student_model_function
from models.teachers.transformer import Transformer as teacher_model_function


PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


def prepare_dataloaders(args, device):
    batch_size = args.batch_size
    data = pickle.load(open(args.data_pkl, 'rb'))

    args.max_token_seq_len = data['settings'].max_len
    args.src_pad_idx = data['vocab']['src'].vocab.stoi[PAD_WORD]  # PAD_WORD='<blank>', src_pad_idx=1
    args.trg_pad_idx = data['vocab']['trg'].vocab.stoi[PAD_WORD]  # 1

    args.src_vocab_size = len(data['vocab']['src'].vocab)  # 9521
    args.trg_vocab_size = len(data['vocab']['trg'].vocab)  # 9521

    if args.embs_share_weight:
        assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, 'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg': data['vocab']['trg']}

    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_iterator, val_iterator


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]  # return the index of the biggest value in each row
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def get_teacher_model(teacher_model_path, args):
    teacher_model_state_dict = torch.load(teacher_model_path, map_location='cpu')['model']
    teacher_model = teacher_model_function(
        args.src_vocab_size,
        args.trg_vocab_size,
        src_pad_idx=args.src_pad_idx,
        trg_pad_idx=args.trg_pad_idx,
        trg_emb_prj_weight_sharing=args.proj_share_weight,
        emb_src_trg_weight_sharing=args.embs_share_weight,
        d_k=args.d_k,
        d_v=args.d_v,
        d_model=args.d_model,
        d_word_vec=args.d_word_vec,
        d_inner=args.d_inner_hid,
        n_layers=args.n_layers,
        n_head=args.n_head,
        dropout=args.dropout,
        scale_emb_or_prj=args.scale_emb_or_prj)
    teacher_model.load_state_dict(teacher_model_state_dict)

    return teacher_model


def get_student_model(teacher_model_path, args, idx, training_periods):
    student_model = student_model_function(
        args.src_vocab_size,
        args.trg_vocab_size,
        src_pad_idx=args.src_pad_idx,
        trg_pad_idx=args.trg_pad_idx,
        trg_emb_prj_weight_sharing=args.proj_share_weight,
        emb_src_trg_weight_sharing=args.embs_share_weight,
        d_k=args.d_k,
        d_v=args.d_v,
        d_model=args.d_model,
        d_word_vec=args.d_word_vec,
        d_inner=args.d_inner_hid,
        n_layers=args.n_layers,
        n_head=args.n_head,
        dropout=args.dropout,
        scale_emb_or_prj=args.scale_emb_or_prj,
        weight_bit=args.weight_bit,
        feature_bit=args.feature_bit,
        fa_res=True,
        is_train=True)
    if idx == 0:
        teacher_model_state_dict = torch.load(teacher_model_path, map_location='cpu')['model']
        student_model_state_dict = student_model.state_dict()
        for k in student_model_state_dict:
            if k in teacher_model_state_dict:
                student_model_state_dict[k] = teacher_model_state_dict[k]
    else:
        student_model_state_dict = torch.load(os.path.join(args.output_dir, 'period_{}_epoch_{}.chkpt'.format(idx-1, training_periods[idx-1])), map_location='cpu')
        student_model_state_dict = student_model_state_dict['model']

    student_model.load_state_dict(student_model_state_dict)

    return student_model


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


def convert_to_onehot(idx_input, vocab_size):
    original_batch, original_word_nums = idx_input.size()
    idx_input = idx_input.reshape(-1, 1).to('cpu')
    one_hot = torch.zeros(idx_input.size(0), vocab_size).scatter(1, idx_input, 1)
    one_hot = one_hot.reshape(original_batch, original_word_nums, vocab_size)
    return one_hot


def get_mse_loss(pred, label, gold, trg_pad_idx):
    assert len(pred) == len(label)
    loss_fn = nn.MSELoss()
    loss = 0
    len_output = len(pred)
    multiplier = torch.logspace(len_output-1, 0, len_output, base=0.8)
    for i in range(len(pred)):
        loss += multiplier[i] * loss_fn(pred[i], label[i])

    non_pad_mask = gold.ne(trg_pad_idx)
    n_word = non_pad_mask.sum().item()

    return loss, n_word


def get_mse_loss_simple(pred, label):
    assert len(pred) == len(label)
    loss_fn = nn.MSELoss()
    loss = 0
    len_output = len(pred)
    multiplier = torch.logspace(len_output-1, 0, len_output, base=0.8)
    for i in range(len(pred)):
        loss += multiplier[i] * loss_fn(pred[i], label[i])

    return loss


def get_cos_loss_simple(pred, label):
    assert len(pred) == len(label)
    loss = 0
    len_output = len(pred)
    multiplier = torch.logspace(len_output-1, 0, len_output, base=0.8)
    for i in range(len(pred)):
        loss += multiplier[i] * (1.0 - torch.mean(torch.cosine_similarity(pred[i], label[i])))

    return loss


def get_loss(pred, label, gold, trg_pad_idx, label_smooth=False):
    if label_smooth:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, label.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = label.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, label, ignore_index=trg_pad_idx, reduction='sum')
    new_pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = new_pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def validate(model, validation_data, device, args):
    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            original_src_seq = patch_src(batch.src, args.src_pad_idx)
            original_trg_seq, gold = patch_trg(batch.trg, args.trg_pad_idx)
            src_seq = convert_to_onehot(original_src_seq, args.src_vocab_size).to(device)
            trg_seq = convert_to_onehot(original_trg_seq, args.trg_vocab_size).to(device)
            original_src_seq = original_src_seq.to(device)
            original_trg_seq = original_trg_seq.to(device)
            gold = gold.to(device)

            # forward
            pred = model(src_seq, trg_seq, original_src_seq, original_trg_seq)
            loss, n_correct, n_word = cal_performance(
                pred, gold, args.trg_pad_idx, smoothing=False)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def print_performances(header, loss, ppl, accu, start_time, lr):
    if ppl is None and accu is None:
        print('  - {header:12} loss: {loss: 8.5f}, lr: {lr:8.5f}, elapse: {elapse:3.3f} min'.format(header=f"({header})", loss=loss, elapse=(time.time()-start_time)/60, lr=lr))
    else:
        print('  - {header:12} loss: {loss: 8.5f}, ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, elapse: {elapse:3.3f} min'.format(header=f"({header})", loss=loss, ppl=ppl, accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))


parser = argparse.ArgumentParser()
parser.add_argument('--weight_bit', type=int, default=2)
parser.add_argument('--feature_bit', type=int, default=4)
parser.add_argument('--translate_direction', type=str, choices=['en2de', 'de2en'], default='en2de')
parser.add_argument('--data_pkl', default='../data/multi30k/m30k_ende_shr.pkl')
parser.add_argument('--teacher_model_path', default='../models/teachers/pretrained_models/transformer_en_de.chkpt')

parser.add_argument('--output_dir', type=str, default='../output_results/archnet_transformer')
parser.add_argument('--use_tb', action='store_true')
parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best')

parser.add_argument('--final_epochs', type=int, default=50)
parser.add_argument('--epochs_per_period', type=int, default=6)
parser.add_argument('--num_of_middle_periods', type=int, default=14)
parser.add_argument('--batch_size', type=int, default=2048)

parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--d_inner_hid', type=int, default=512)
parser.add_argument('--d_k', type=int, default=64)
parser.add_argument('--d_v', type=int, default=64)

parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--n_layers', type=int, default=6)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--seed', type=int, default=None)

parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--embs_share_weight', action='store_true')
parser.add_argument('--proj_share_weight', action='store_true')
parser.add_argument('--scale_emb_or_prj', type=str, default='prj')

parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--label_smoothing', action='store_true')

if __name__ == "__main__":
    args = parser.parse_args()
    args.output_dir = args.output_dir + "_{}_{}w{}a".format(args.translate_direction, args.weight_bit, args.feature_bit)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    args.d_word_vec = args.d_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
    if args.use_tb:
        print("[Info] Use Tensorboard")
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))

    # set training_periods
    training_periods = [args.epochs_per_period for _ in range(args.num_of_middle_periods)] + [args.final_epochs]
    for idx, epochs in enumerate(training_periods):

        if args.data_pkl:
            training_data, validation_data = prepare_dataloaders(args, device)
        else:
            raise

        teacher_model = get_teacher_model(args.teacher_model_path, args)
        teacher_model = teacher_model.to(device)
        teacher_model.eval()

        student_model = get_student_model(args.teacher_model_path, args, idx, training_periods)
        student_model = student_model.to(device)
        student_model.train()

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, student_model.parameters()), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-9)
        if idx == len(training_periods) - 1:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-4)

        if idx == 0:
            log_train_file = os.path.join(args.output_dir, 'train.log')
            log_valid_file = os.path.join(args.output_dir, 'valid.log')

            print('[Info] Training performance will be written to file: {} and {}'.format(
                log_train_file, log_valid_file))

            with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
                log_tf.write('epoch,loss,ppl,accuracy\n')
                log_vf.write('epoch,loss,ppl,accuracy\n')

        # valid_accus = []
        valid_losses = []
        valid_accus = []
        valid_ppls = []
        total_loss, n_word_total, n_word_correct = 0, 0, 0

        desc = '  - (Training)   '

        for epoch in range(epochs):
            print('[ Period/Epoch', idx, '/', epoch, ']')
            start = time.time()
            for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):

                # prepare data
                teacher_src_seq = patch_src(batch.src, args.src_pad_idx)
                teacher_trg_seq, gold = patch_trg(batch.trg, args.trg_pad_idx)
                student_src_seq = convert_to_onehot(teacher_src_seq, args.src_vocab_size).to(device)
                student_trg_seq = convert_to_onehot(teacher_trg_seq, args.trg_vocab_size).to(device)
                teacher_src_seq = teacher_src_seq.to(device)
                teacher_trg_seq = teacher_trg_seq.to(device)
                gold = gold.to(device)

                with torch.no_grad():
                    label = teacher_model(teacher_src_seq, teacher_trg_seq)
                    label = label.max(1)[1]

                # forward
                optimizer.zero_grad()
                pred = student_model(student_src_seq, student_trg_seq, teacher_src_seq, teacher_trg_seq)

                # backward and update parameters
                if idx == 0:
                    loss1, n_word = get_mse_loss(student_model.encoder.middle_outputs['encoder_outputs'][:1], teacher_model.encoder.middle_outputs['encoder_outputs'][:1], gold, args.trg_pad_idx)
                    loss2 = get_cos_loss_simple(student_model.encoder.middle_outputs['embedding_outputs'], teacher_model.encoder.middle_outputs['embedding_outputs'])
                    loss = loss1 + loss2
                elif idx < 7:
                    loss1, n_word = get_mse_loss(student_model.encoder.middle_outputs['encoder_outputs'][:idx*2+1], teacher_model.encoder.middle_outputs['encoder_outputs'][:idx*2+1], gold, args.trg_pad_idx)
                    loss2 = get_cos_loss_simple(student_model.encoder.middle_outputs['encoder_attention_outputs'][:idx], teacher_model.encoder.middle_outputs['encoder_attention_outputs'][:idx])
                    loss3 = get_cos_loss_simple(student_model.encoder.middle_outputs['embedding_outputs'], teacher_model.encoder.middle_outputs['embedding_outputs'])
                    loss = loss1 + loss2 + loss3
                elif idx == 7:
                    loss1, n_word = get_mse_loss(student_model.encoder.middle_outputs['encoder_outputs'], teacher_model.encoder.middle_outputs['encoder_outputs'], gold, args.trg_pad_idx)
                    loss2 = get_cos_loss_simple(student_model.encoder.middle_outputs['encoder_attention_outputs'], teacher_model.encoder.middle_outputs['encoder_attention_outputs'])
                    loss3 = get_cos_loss_simple(student_model.encoder.middle_outputs['embedding_outputs'], teacher_model.encoder.middle_outputs['embedding_outputs'])
                    loss4 = get_mse_loss_simple(student_model.decoder.middle_outputs['decoder_outputs'][:1], teacher_model.decoder.middle_outputs['decoder_outputs'][:1])
                    loss5 = get_cos_loss_simple(student_model.decoder.middle_outputs['embedding_outputs'], teacher_model.decoder.middle_outputs['embedding_outputs'])
                    loss = loss1 + loss2 + loss3 + loss4 + loss5
                elif idx < len(training_periods) - 1:
                    loss1, n_word = get_mse_loss(student_model.encoder.middle_outputs['encoder_outputs'], teacher_model.encoder.middle_outputs['encoder_outputs'], gold, args.trg_pad_idx)
                    loss2 = get_cos_loss_simple(student_model.encoder.middle_outputs['encoder_attention_outputs'], teacher_model.encoder.middle_outputs['encoder_attention_outputs'])
                    loss3 = get_cos_loss_simple(student_model.encoder.middle_outputs['embedding_outputs'], teacher_model.encoder.middle_outputs['embedding_outputs'])
                    loss4 = get_mse_loss_simple(student_model.decoder.middle_outputs['decoder_outputs'][:(idx-7)*3+1], teacher_model.decoder.middle_outputs['decoder_outputs'][:(idx-7)*3+1])
                    loss5 = get_cos_loss_simple(student_model.decoder.middle_outputs['decoder_attention_outputs'][:idx-7], teacher_model.decoder.middle_outputs['decoder_attention_outputs'][:idx-7])
                    loss6 = get_cos_loss_simple(student_model.decoder.middle_outputs['decoder_encoder_attention_outputs'][:idx-7], teacher_model.decoder.middle_outputs['decoder_encoder_attention_outputs'][:idx-7])
                    loss7 = get_cos_loss_simple(student_model.decoder.middle_outputs['embedding_outputs'], teacher_model.decoder.middle_outputs['embedding_outputs'])
                    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
                else:
                    loss1, n_correct, n_word = get_loss(pred, label, gold, args.trg_pad_idx, label_smooth=True)
                    loss2 = get_mse_loss_simple(student_model.encoder.middle_outputs['encoder_outputs'], teacher_model.encoder.middle_outputs['encoder_outputs'])
                    loss3 = get_mse_loss_simple(student_model.decoder.middle_outputs['decoder_outputs'], teacher_model.decoder.middle_outputs['decoder_outputs'])
                    loss4 = get_cos_loss_simple(student_model.encoder.middle_outputs['embedding_outputs'], teacher_model.encoder.middle_outputs['embedding_outputs'])
                    loss5 = get_cos_loss_simple(student_model.decoder.middle_outputs['embedding_outputs'], teacher_model.decoder.middle_outputs['embedding_outputs'])
                    loss6 = get_cos_loss_simple(student_model.encoder.middle_outputs['encoder_attention_outputs'], teacher_model.encoder.middle_outputs['encoder_attention_outputs'])
                    loss7 = get_cos_loss_simple(student_model.decoder.middle_outputs['decoder_attention_outputs'], teacher_model.decoder.middle_outputs['decoder_attention_outputs'])
                    loss8 = get_cos_loss_simple(student_model.decoder.middle_outputs['decoder_encoder_attention_outputs'], teacher_model.decoder.middle_outputs['decoder_encoder_attention_outputs'])
                    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
                    n_word_correct += n_correct
                loss.backward()
                optimizer.step()

                # note keeping
                n_word_total += n_word
                total_loss += loss.item()

            if idx < len(training_periods) - 1:
                lr = optimizer.param_groups[0]['lr']
                train_loss = total_loss / n_word_total
                print_performances('Training', train_loss, None, None, start, lr)
                if epoch == epochs - 1:
                    checkpoint = {'period': idx, 'epoch': epoch, 'settings': args, 'model': student_model.state_dict()}
                    model_name = 'period_{}_epoch_{}.chkpt'.format(idx, epoch+1)
                    torch.save(checkpoint, os.path.join(args.output_dir, model_name))
                    if idx > 0:
                        delete_model_path = os.path.join(args.output_dir, 'period_{}_epoch_{}.chkpt'.format(idx-1, training_periods[idx-1]))
                        if os.path.exists(delete_model_path):
                            os.remove(delete_model_path)
            else:
                scheduler.step()
                train_loss = total_loss/n_word_total
                train_accuracy = n_word_correct/n_word_total

                train_ppl = math.exp(min(train_loss, 100))
                # Current learning rate
                lr = optimizer.param_groups[0]['lr']
                print_performances('Training', train_loss, train_ppl, train_accuracy, start, lr)

                start = time.time()
                valid_loss, valid_accu = validate(student_model, validation_data, device, args)
                valid_ppl = math.exp(min(valid_loss, 100))
                print_performances('Validation', valid_loss, valid_ppl, valid_accu, start, lr)
                valid_losses += [valid_loss]
                valid_accus += [valid_accu]
                valid_ppls += [valid_ppl]

                # save and other settings
                checkpoint = {'epoch': epoch, 'settings': args, 'model': student_model.state_dict()}

                if args.save_mode == 'all':
                    model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                    torch.save(checkpoint, model_name)
                elif args.save_mode == 'best':
                    model_name_min_loss = 'model_min_loss.chkpt'
                    model_name_max_acc = 'model_max_acc.chkpt'
                    if valid_loss <= min(valid_losses):
                        torch.save(checkpoint, os.path.join(args.output_dir, model_name_min_loss))
                        print('    - [Info] The checkpoint file has been updated - min valid loss.')
                    if valid_accu >= max(valid_accus):
                        torch.save(checkpoint, os.path.join(args.output_dir, model_name_max_acc))
                        print('    - [Info] The checkpoint file has been updated - max valid accuracy')
                with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                    log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                        epoch=epoch, loss=train_loss,
                        ppl=train_ppl, accu=100*train_accuracy))
                    log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                        epoch=epoch, loss=valid_loss,
                        ppl=valid_ppl, accu=100*valid_accu))

                if args.use_tb:
                    tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch)
                    tb_writer.add_scalars('accuracy', {'train': train_accuracy*100, 'val': valid_accu*100}, epoch)
                    tb_writer.add_scalar('learning_rate', lr, epoch)

    print('max_valid_acc_epoch: {}, max_valid_acc: {}, max_valid_acc_ppl: {}, max_valid_acc_loss: {}'.format(valid_accus.index(max(valid_accus)), max(valid_accus), valid_ppls[valid_accus.index(max(valid_accus))], valid_losses[valid_accus.index(max(valid_accus))]))
    print('Finish Training')
