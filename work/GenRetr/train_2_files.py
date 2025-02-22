#coding=utf-8
"""
    run for generation by unimo-text
    GenRet
"""
import argparse
import os
import time
import json
from math import ceil

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from gen_utils import prepare_data_loader, print_args, select_sum, set_seed
from paddle.optimizer import AdamW
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import BLEU
from paddlenlp.transformers import (
    BasicTokenizer,
    LinearDecayWithWarmup,
    UNIMOLMHeadModel,
    UNIMOTokenizer,
)

# yapf: disable
def parse_args():
    """
        parse args
    """
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model_name_or_path', type=str, default='/mlx_devbox/users/lianzetao/playground/person/test2/work/unimo-text-1.0', \
            help='The path or shortcut name of the pre-trained model.')
    parser.add_argument("--train_file1", type=str, required=True, default=None, help="Train data path1.")
    parser.add_argument("--train_file2", type=str, required=True, default=None, help="Train data path2.")
    parser.add_argument("--ratio1", type=int, required=True, default=None, help="Train data ratio1.")
    parser.add_argument("--ratio2", type=int, required=True, default=None, help="Train data ratio2.")
    parser.add_argument("--dev_file", type=str, required=False, default=None, help="Dev data path.")
    parser.add_argument("--test_file", type=str, required=False, default=None, help="Test data path.")
    parser.add_argument('--save_dir', type=str, default='./model_checkpoints',
                        help='The directory where the checkpoints will be saved.')
    parser.add_argument('--logging_steps', type=int, default=100, help='Log every X updates steps.')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save checkpoint every X updates steps.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for initialization.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU/CPU for training.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='The initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='The weight decay for optimizer.')
    parser.add_argument('--epochs', type=int, default=3, help='Total number of training epochs to perform.')
    parser.add_argument('--warmup_proportion', type=float, default=0.02, help='The number of warmup steps.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='The max value of grad norm.')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
    parser.add_argument('--beta2', type=float, default=0.98, help='beta2')
    parser.add_argument('--epsilon', type=float, default=1e-6, help='epsilon')
    parser.add_argument('--max_seq_len', type=int, default=512, help='The maximum sequence length of training.')
    parser.add_argument('--max_dec_len', type=int, default=20, help='The maximum sequence length of decoding.')
    parser.add_argument('--min_dec_len', type=int, default=3, help='The minimal sequence length of decoding.')
    parser.add_argument('--max_target_len', type=int, default=6,
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
    parser.add_argument('--length_penalty', type=float, default=1.2, \
            help='The exponential penalty to the sequence length for beam search.')
    parser.add_argument('--device', type=str, default='gpu', help='The device to select for training the model.')
    parser.add_argument('--output_path', type=str, default='./predict.txt', \
            help='The file path where the infer result will be saved.')
    parser.add_argument("--do_train", action='store_true', default=False,  help="Whether to train the model.")
    parser.add_argument("--do_eval", action='store_true', default=False, help="Whether to eval and predict.")
    parser.add_argument("--do_test", action='store_true', default=False, help="Whether to eval and predict.")

    args = parser.parse_args()
    return args
# yapf: enable


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


def save_ckpt(model, tokenizer, save_dir, name):
    """
        save checkpoint
    """
    output_dir = os.path.join(save_dir, "model_{}".format(name))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Need better way to get inner model of DataParallel
    model_to_save = model._layers if isinstance(model, paddle.DataParallel) else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# Multi-task learning dataloder
def custom_alternating_dataloader(dataloader1, dataloader2, ratio1, ratio2, max_epochs = 1):
    iter1 = iter(dataloader1)
    iter2 = iter(dataloader2)

    epoch = 1

    while True:
        for _ in range(ratio1):
            try:
                yield next(iter1)
            except StopIteration:
                iter1 = iter(dataloader1)
                yield next(iter1)
        
        for _ in range(ratio2):
            try:
                yield next(iter2)
            except StopIteration:
                if max_epochs is not None and epoch >= max_epochs:
                    return
                iter2 = iter(dataloader2)
                epoch += 1
                yield next(iter2)


def run(args):
    paddle.set_device(args.device)
    world_size = dist.get_world_size()
    print("world_size:", world_size)

    if world_size > 1:
        dist.init_parallel_env()

    model = UNIMOLMHeadModel.from_pretrained(args.model_name_or_path)
    tokenizer = UNIMOTokenizer.from_pretrained(args.model_name_or_path)

    if world_size > 1:
        model = paddle.DataParallel(model)

    if args.train_file1 and args.train_file2:
        train_ds1, train_data_loader1 = prepare_data_loader(args.train_file1, tokenizer, args, "train")
        train_ds2, train_data_loader2 = prepare_data_loader(args.train_file2, tokenizer, args, "train")
        alternating_loader = custom_alternating_dataloader(train_data_loader1, train_data_loader2, ratio1 = args.ratio1, ratio2 = args.ratio2, max_epochs = args.epochs)
    else:
        raise ValueError("Both train_file1 and train_file2 must be provided.")

    if args.dev_file:
        dev_ds, dev_data_loader = prepare_data_loader(args.dev_file, tokenizer, args, "test")

    if args.test_file:
        test_ds, test_data_loader = prepare_data_loader(args.test_file, tokenizer, args, "test")

    if args.do_train:
        print("in training process")
        model.train()

        # The total number of training steps
        num_training_steps = ceil(args.epochs * len(train_data_loader) * args.ratio1 / args.ratio2) + args.epochs * len(train_data_loader)
        lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, args.warmup_proportion)

        decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]

        optimizer = AdamW(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.epsilon,
            apply_decay_param_fun=lambda x: x in decay_params,
            grad_clip=paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm),
        )

        step = 0
        total_time = 0.0

        for epoch in range(args.epochs):
            print("\nEpoch %d/%d, train data size: %d" % (epoch + 1, args.epochs, ceil(num_training_steps / args.epochs)))
            batch_start_time = time.time()
            for _ in range(ceil(num_training_steps / args.epochs)):
                step += 1
                try:
                    inputs = next(alternating_loader)
                except StopIteration:
                    print("All data has been processed.")
                    if dist.get_rank() == 0:
                        save_ckpt(model, tokenizer, args.save_dir, f"step_{step}_end")
                        print(f"Saved model at step {step}.\n")
                    return

                labels = inputs[-1]
                logits = model(*inputs[:-1])
                labels = paddle.nn.functional.one_hot(labels, num_classes=logits.shape[-1])
                labels = paddle.nn.functional.label_smooth(labels)
                loss = F.cross_entropy(logits, labels, soft_label=True)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

                total_time += time.time() - batch_start_time
                if step % args.logging_steps == 0:
                    ppl = paddle.exp(loss)
                    print("step %d - loss: %.4f - ppl: %.4f - lr: %.7f - %.3fs/step"
                        % (step, loss, ppl, optimizer.get_lr(), total_time / args.logging_steps)
                    )
                    total_time = 0.0
                if step % args.save_steps == 0:
                    print("will save model")
                    if dist.get_rank() == 0:
                        save_ckpt(model, tokenizer, args.save_dir, step)
                        print("Saved step {} model.\n".format(step))

                batch_start_time = time.time()

            print("will save model")
            if dist.get_rank() == 0:
                save_ckpt(model, tokenizer, args.save_dir, f"epoch_{epoch}")
                print(f"Saved epoch {epoch} model.\n")
                if args.do_eval:
                    model_eval = model._layers if isinstance(model, paddle.DataParallel) else model
                    evaluation(model_eval, dev_data_loader, args, tokenizer)

        print("\nTraining completed.")
    elif args.do_test:
        model_eval = model._layers if isinstance(model, paddle.DataParallel) else model
        evaluation(model_eval, test_data_loader, args, tokenizer)
        print("\nTest completed.")


@paddle.no_grad()
def evaluation(model, data_loader, args, tokenizer):
    """
        predict and  gen res
    """
    print("\nEval begin...")
    model.eval()
    pred_ref = []
    total_time = 0.0
    start_time = time.time()
    count_ids = 0
    for step, inputs in enumerate(data_loader, 1):
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
        print(results)
        pred_ref.extend(results)
        start_time = time.time()

    print(count_ids)

    with open(args.output_path, "w", encoding="utf-8") as fout:
        for ref in pred_ref:
            fout.write(ref + "\n")

    print("\nSave inference result into: %s" % args.output_path)

    if "target" in data_loader.dataset[0].keys():
        targets = [example["target"] for example in data_loader.dataset]
        calc_bleu(pred_ref, targets)

    model.train()
    return


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    run(args)
