#coding=utf-8
"""
    run for generation by unimo-text
    GenRet
"""
import argparse
import os
import time
import json

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from gen_utils import prepare_data_loader, print_args, select_sum, set_seed
from paddle.optimizer import AdamW
from paddlenlp.metrics import BLEU
from paddlenlp.transformers import (
    BasicTokenizer,
    LinearDecayWithWarmup,
    UNIMOLMHeadModel,
    UNIMOTokenizer,
)

def parse_args():
    """
        parse args
    """
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model_name_or_path', type=str, default='unimo-text-1.0', \
            help='The path or shortcut name of the pre-trained model.')
    parser.add_argument("--params_path", type=str, required=True,
                    help="The path to model parameters to be loaded.")
    parser.add_argument("--test_file", type=str, required=False, default=None, help="Test data path.")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size per GPU/CPU for training.')
    parser.add_argument('--max_seq_len', type=int, default=512, help='The maximum sequence length of training.')
    parser.add_argument('--max_dec_len', type=int, default=20, help='The maximum sequence length of decoding.')
    parser.add_argument('--min_dec_len', type=int, default=3, help='The minimal sequence length of decoding.')
    parser.add_argument('--max_target_len', type=int, default=30,
                        help='The maximum target sequence length of training.')
    parser.add_argument('--max_title_len', type=int, default=30, help='The maximum title sequence length of training.')
    parser.add_argument('--num_return_sequences', type=int, default=1, \
            help='The numbers of returned sequences for one input in generation.')
    parser.add_argument('--decode_strategy', type=str, default='beam_search', help='The decode strategy in generation.')
    parser.add_argument('--top_k', type=int, default=0, \
            help='The number of highest probability vocabulary tokens to keep for top-k sampling.')
    parser.add_argument('--temperature', type=float, default=1.0, \
            help='The value used to module the next token probabilities.')
    parser.add_argument('--top_p', type=float, default=1.0, help='The cumulative probability for top-p sampling.')
    parser.add_argument('--num_beams', type=int, default=6, help='The number of beams for beam search.')
    parser.add_argument('--logging_steps', type=int, default=100, help='Log every X updates steps.')
    parser.add_argument('--length_penalty', type=float, default=1.2, \
            help='The exponential penalty to the sequence length for beam search.')
    parser.add_argument('--device', type=str, default='gpu', help='The device to select for training the model.')
    parser.add_argument('--output_path', type=str, default='./predict.txt', \
            help='The file path where the infer result will be saved.')

    args = parser.parse_args()
    return args


def calc_bleu(preds, targets):
    """
        calculate bleu
    """
    assert len(preds) == len(targets), (
        "The length of pred_responses should be equal to the length of "
        "target_responses. But received {} and {}.".format(len(preds), len(targets))
    )
    bleu4 = BLEU(n_size=4)
    tokenizer = BasicTokenizer()

    for pred, target in zip(preds, targets):
        pred_tokens = tokenizer.tokenize(pred)
        target_token = tokenizer.tokenize(target)

        bleu4.add_inst(pred_tokens, [target_token])

    print("\n" + "*" * 15)
    print("The auto evaluation result is:")
    print("BLEU-4:", bleu4.score())


def run(args):
    """
        run train
    """
    paddle.set_device(args.device)

    model = UNIMOLMHeadModel.from_pretrained(args.model_name_or_path)
    tokenizer = UNIMOTokenizer.from_pretrained(args.model_name_or_path)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    else:
        raise ValueError("Please set --params_path with correct pretrained model file")


    test_ds, test_data_loader = prepare_data_loader(args.test_file, tokenizer, args, "test")
    #model_eval = model._layers if isinstance(model, paddle.DataParallel) else model

    model.eval()
    pred_ref = []
    total_time = 0.0
    start_time = time.time()
    count_ids = 0
    for step, inputs in enumerate(test_data_loader, 1):
        input_ids, token_type_ids, position_ids, attention_mask = inputs
        ids, scores = model.generate(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            max_length=args.max_dec_len,
            min_length=args.min_dec_len,
            decode_strategy=args.decode_strategy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            num_return_sequences=args.num_return_sequences,
            bos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.mask_token_id,
        )
        count_ids += ids.shape[0]

        total_time += time.time() - start_time
        if step % args.logging_steps == 0:
            print("step %d - %.3fs/step" % (step, total_time / args.logging_steps))
            total_time = 0.0

        results = select_sum(ids, scores, tokenizer, args.max_dec_len, args.num_return_sequences)
        #print(results)
        pred_ref.extend(results)
        start_time = time.time()

    print(count_ids)

    with open(args.output_path, "w", encoding="utf-8") as fout:
        for ref in pred_ref:
            fout.write(ref + "\n")

    print("\nSave inference result into: %s" % args.output_path)

    return


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    run(args)
