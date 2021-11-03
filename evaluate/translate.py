''' Translate input text with trained model. '''
import torch
import argparse
import dill as pickle
from tqdm import tqdm
import os

from torchtext.data import Dataset
from translator import Translator
import sys
sys.path.append("..")
from models.students.transformer import ArchNetTransformer as Transformer


PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


def convert_to_onehot(idx_input, vocab_size):
    original_batch, original_word_nums = idx_input.size()
    idx_input = idx_input.reshape(-1, 1).to('cpu')
    one_hot = torch.zeros(idx_input.size(0), vocab_size).scatter(1, idx_input, 1)
    one_hot = one_hot.reshape(original_batch, original_word_nums, vocab_size)
    return one_hot


def load_model(opt, device):

    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']

    model = Transformer(
        model_opt.src_vocab_size,
        model_opt.trg_vocab_size,

        model_opt.src_pad_idx,
        model_opt.trg_pad_idx,

        trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_trg_weight_sharing=model_opt.embs_share_weight,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout,
        weight_bit=model_opt.weight_bit,
        feature_bit=model_opt.feature_bit,
        is_train=False).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model, model_opt.output_dir


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('--model', required=True, help='Path to model weight file')
    parser.add_argument('--data_pkl', required=True, help='Pickle file with both instances and vocabulary.')
    # parser.add_argument('--output', default='otuput/pred.txt', help="""Path to output the predictions (each line will be the decoded sequence""")
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--max_seq_len', type=int, default=100)
    parser.add_argument('--no_cuda', action='store_true')
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    data = pickle.load(open(opt.data_pkl, 'rb'))
    SRC, TRG = data['vocab']['src'], data['vocab']['trg']
    opt.src_pad_idx = SRC.vocab.stoi[PAD_WORD]
    opt.trg_pad_idx = TRG.vocab.stoi[PAD_WORD]
    opt.trg_bos_idx = TRG.vocab.stoi[BOS_WORD]
    opt.trg_eos_idx = TRG.vocab.stoi[EOS_WORD]
    opt.src_vocab_size = len(SRC.vocab)
    opt.trg_vocab_size = len(TRG.vocab)

    test_loader = Dataset(examples=data['test'], fields={'src': SRC, 'trg': TRG})

    device = torch.device('cuda' if opt.cuda else 'cpu')
    model, output_dir = load_model(opt, device)
    translator = Translator(
        model=model,
        beam_size=opt.beam_size,
        max_seq_len=opt.max_seq_len,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_bos_idx=opt.trg_bos_idx,
        trg_eos_idx=opt.trg_eos_idx,
        trg_vocab_size=opt.trg_vocab_size).to(device)

    unk_idx = SRC.vocab.stoi[SRC.unk_token]
    with open(os.path.join(output_dir, 'predictions.txt'), 'w') as f:
        for example in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            original_src_seq = torch.LongTensor([[SRC.vocab.stoi.get(word, unk_idx) for word in example.src]]).to(device)
            onehot_src_seq = convert_to_onehot(original_src_seq, opt.src_vocab_size).to(device)
            pred_seq = translator.translate_sentence(onehot_src_seq, original_src_seq)
            pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
            pred_line = pred_line.replace(BOS_WORD, '').replace(EOS_WORD, '')
            pred_line = pred_line.replace(' ,', ',')
            pred_line = pred_line.replace(' .', '.')
            f.write(pred_line.strip() + '\n')

    print('[Info] Finished.')


if __name__ == "__main__":
    main()
