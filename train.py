"""
Description: Code for benchmarking different methods of cross-lingual knowledge transfer,
including: 1) multilingual encoders, 2) hard translation; 3) soft (latent) translation.

Author: Edoardo M. Ponti
"""

import argparse
import logging
import os
import random
import csv
import json
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (
    AdamW,
    XLMRobertaConfig, RobertaConfig,
    XLMRobertaTokenizer, RobertaTokenizer,
    XLMRobertaForMultipleChoice, XLMRobertaForSequenceClassification,
    RobertaForMultipleChoice, RobertaForSequenceClassification,
    get_linear_schedule_with_warmup, MarianTokenizer, MarianMTModel,
    MBartForConditionalGeneration, MBart50TokenizerFast,
    MBartForSequenceClassification, MBartConfig,
    M2M100Config, M2M100Tokenizer, M2M100ForConditionalGeneration,
    XLMRobertaForQuestionAnswering, RobertaForQuestionAnswering,
    MBartForQuestionAnswering,
)
from transformers.hf_api import HfApi
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate
from transformers.data.processors.squad import squad_convert_examples_to_features, SquadResult

from utils import (
    convert_examples_to_features, convert_examples_MC_to_features,
    SCExample, MCExample, TiDiQAProcessor,
)
from custom import (
    TranslatorClassifier,
    MBartForMultipleChoice, M2M100ForMultipleChoice,
    generate_with_grad,
)
from reranker import RobertaForReranking


logger = logging.getLogger(__name__)

task2eval_languages = {'pawsx': ["de", "en", "es", "fr", "ja", "ko", "zh"],
                       'xnli': ["ar", "bg", "de", "el", "en", "es", "fr", "hi",
                                "ru", "sw", "th", "tr", "ur", "vi", "zh"],
                       'xcopa': ["en", "et", "ht", "id", "it", "qu", "sw", "ta",
                                 "th", "tr", "vi", "zh"],
                       'tydiqa': ["ar", "bn", "en", "fi", "id", "ko", "ru", "sw", "te"]
                       }

mode2eval_languages = {'multi': ["ht", "qu"],
                       'trans-hard': [],
                       'trans-soft': ["en"],
                       }

task2labels = {'pawsx': ["0", "1"],
               'xnli': ["contradiction", "entailment", "neutral"],
               'xcopa': [None],
               'tydiqa': [None, None],
               }


def compute_metrics(preds, labels):
    scores = {
        "acc": (preds == labels).mean(),
        "num": len(
            preds),
        "correct": (preds == labels).sum()
    }
    return scores


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """Train the model."""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("    Num examples = %d", len(train_dataset))
    logger.info("    Num Epochs = %d", args.num_train_epochs)
    logger.info("    Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "    Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("    Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("    Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    best_score = 0
    best_checkpoint = None
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, args.num_train_epochs, desc="Epoch", disable=True
    )
    set_seed(args)    # Added here for reproductibility
    for epoch_n in train_iterator:
        logger.info("Epoch: %s", epoch_n)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        for step, batch in enumerate(epoch_iterator):

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            if args.cls_model_name not in ['mbart50', 'm2m100', 'mt5']:
                inputs["token_type_ids"] = batch[2]
            if args.task_name == "tydiqa":
                inputs.update({"start_positions": batch[3], "end_positions": batch[4]})
            else:
                inputs.update({"labels": batch[3]})
            outputs = model(**inputs)
            loss = outputs.loss

            if args.n_gpu > 1:
                loss = loss.mean()    # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()    # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("Training NLL: %s", (tr_loss - logging_loss) / args.logging_steps)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    result = evaluate(args, model, tokenizer, split='dev',
                                      language=args.train_language, prefix=str(global_step))
                    metric = 'acc' if args.task_name != "tydiqa" else 'exact'
                    logger.info(" Dev accuracy {} = {}".format(args.train_language, result[metric]))
                    if result[metric] > best_score:
                        logger.info(" result={} > best_score={}".format(result[metric], best_score))
                        output_dir = os.path.join(args.output_dir, "checkpoint-best")
                        best_checkpoint = output_dir
                        best_score = result[metric]
                        # Save model checkpoint
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )    # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_score, best_checkpoint


def evaluate(args, model, tokenizer, split='train', language='en', prefix="",
             output_file=None, label_list=None, output_only_prediction=True, prepare_trans=False):
    """Evalute the model."""
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_all = load_and_cache_examples(args, eval_task, tokenizer, split=split,
                                           language=language, prepare_trans=prepare_trans,
                                           get_features=True)
        eval_dataset, eval_examples, eval_features = eval_all

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} {} *****".format(prefix, language))
        logger.info("    Num examples = %d", len(eval_dataset))
        logger.info("    Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        sentences = None
        qa_results = []
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=True):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                if args.cls_model_name not in ['mbart50', 'm2m100', 'mt5']:
                    inputs["token_type_ids"] = batch[2]
                if args.task_name == "tydiqa":
                    example_indices = batch[3]
                else:
                    inputs.update({"labels": batch[3]})
                outputs = model(**inputs)

            nb_eval_steps += 1

            if args.task_name != "tydiqa":
                tmp_eval_loss, logits = outputs.loss, outputs.logits
                eval_loss += tmp_eval_loss.mean().item()

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                    sentences = inputs["input_ids"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                    sentences = np.append(sentences, inputs["input_ids"].detach().cpu().numpy(), axis=0)
            else:
                for example_index, sl, el in zip(example_indices, outputs.start_logits, outputs.end_logits):
                    eval_feature = eval_features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)

                    start_logits = sl.detach().cpu().tolist()
                    end_logits = el.detach().cpu().tolist()

                    result = SquadResult(unique_id, start_logits, end_logits)
                    qa_results.append(result)

        if args.task_name != "tydiqa":
            preds = np.argmax(preds, axis=1)
            result = compute_metrics(preds, out_label_ids)
            results.update(result)
        else:
            predictions = compute_predictions_logits(
                eval_examples,
                eval_features,
                qa_results,
                args.n_best_size,
                args.max_answer_length,
                False,
                None,
                None,
                None,
                args.verbose_logging,
                False,
                args.null_score_diff_threshold,
                tokenizer,
            )

            # Compute the F1 and exact scores.
            results = squad_evaluate(eval_examples, predictions)

        logger.info("***** Eval results {} {} *****".format(prefix, language))
        for key in sorted(results.keys()):
            logger.info("    %s = %s", key, str(results[key]))

    return results


def refine(args, train_dataset, model, tokenizer, language, prepare_trans=False):
    """Train the model."""

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'translator' not in n],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'translator' not in n], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.mode == 'trans-soft':
        translator_optimizer = torch.optim.SGD(model.translator.parameters(), lr=1e-2)

    # Train!
    logger.info("***** Running refinement *****")
    logger.info("    Num examples = %d", len(train_dataset))
    logger.info("    Num Epochs = %d", args.num_train_epochs)
    logger.info("    Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "    Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps,
    )
    logger.info("    Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("    Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    result_test = None
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, args.num_train_epochs, desc="Epoch", disable=True
    )
    set_seed(args)    # Added here for reproductibility
    for epoch_n in train_iterator:
        logger.info("Epoch: {}".format(epoch_n))
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        for step, batch in enumerate(epoch_iterator):

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            if args.cls_model_name not in ['mbart50', 'm2m100', 'mt5']:
                inputs["token_type_ids"] = batch[2]
            if args.task_name == "tydiqa":
                inputs.update({"start_positions": batch[3], "end_positions": batch[4]})
            else:
                inputs.update({"labels": batch[3]})
            outputs = model(**inputs)
            loss = outputs.loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if args.mode == 'trans-soft':
                nmt_loss = outputs.nmt_loss / args.gradient_accumulation_steps
                nmt_loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()    # Update learning rate schedule
                if args.mode == 'trans-soft':
                    translator_optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    logger.info("Training loglik: {}".format((tr_loss - logging_loss) / args.logging_steps))
                    logging_loss = tr_loss

    if args.local_rank in [-1, 0]:
        result_test = evaluate(args, model, tokenizer, split='test', language=language, prefix=str(global_step), prepare_trans=prepare_trans)

        if args.do_analyse:
            translations_file = os.path.join(args.output_dir, 'translations_mrt.json')
            with open(translations_file, 'a', encoding="utf-8") as writer:
                analyse(args, model, tokenizer, split='dev',
                        language=language, prefix='best_checkpoint',
                        writer=writer, logits_matrix=None)

    return global_step, tr_loss / global_step, result_test


def analyse(args, model, tokenizer, split='train', language='en', prefix='',
            writer=None, label_list=None, logits_matrix=None):
    """Analyse the model."""
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, split=split,
                                               language=language, prepare_trans=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running analysis {} {} *****".format(prefix, language))
        logger.info("    Num examples = %d", len(eval_dataset))
        logger.info("    Batch size = %d", args.eval_batch_size)

        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        sentences = None
        scores = None
        translations = []

        for batch in tqdm(eval_dataloader, desc="Analysing", disable=True):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                if args.cls_model_name not in ['mbart50', 'm2m100', 'mt5']:
                    inputs["token_type_ids"] = batch[2]
                inputs.update({"labels": batch[3]})
                outputs = model(**inputs)
                translation, logits, score = outputs.translations, outputs.all_logits, outputs.scores

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
                sentences = inputs["input_ids"].detach().cpu().numpy()
                scores = score.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                sentences = np.append(sentences, inputs["input_ids"].detach().cpu().numpy(), axis=0)
                scores = np.append(scores, score.detach().cpu().numpy(), axis=0)
            translations.extend(translation)

        max_preds = np.argmax(preds, axis=-1)
        max_ensemble = np.argmax(np.mean(preds, axis=1), axis=-1)

        print(len(list(max_ensemble)), len(list(max_preds)), len(list(out_label_ids)), len(sentences), len(translations))

        for e, p, lb, s, t, sc in zip(list(max_ensemble), list(max_preds), list(out_label_ids), sentences, translations, scores.tolist()):
            s = [tokenizer.decode(sent, skip_special_tokens=True).replace("▁", " ") for sent in s]
            p = list(map(int, p))
            e, lb = int(e), int(lb)
            if label_list:
                e = label_list.get(e, e)
                lb = label_list.get(lb, lb)
            line = json.dumps({"language": language, "sentence": s, "label": lb, "translations": t,
                               "predictions": p, "ensemble_prediction": e, "scores": sc}, ensure_ascii=False)
            writer.write(line + "\n")

        if logits_matrix is not None:
            logits_matrix[language] = preds


def get_examples_SC(data_dir, language='en', split='train'):

    def _read_tsv(input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=None))

    examples = []
    lines = _read_tsv(os.path.join(data_dir, "{}-{}.tsv".format(split, language)))

    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s-%s" % (split, language, i)
        text_a = line[0]
        text_b = line[1]
        label = str(line[2].strip())
        assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
        examples.append(SCExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=language))
    return examples


def get_examples_MC(data_dir, language='en', split='train'):

    def _read_jsonl(input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            lines = []
            for line in f:
                line = json.loads(line)
                lines.append(line)
            return lines

    examples = []
    lines = _read_jsonl(os.path.join(data_dir, "{}.{}.jsonl".format(split, language)))

    for (i, line) in enumerate(lines):
        guid = "%s-%s-%s" % (split, language, i)
        contexts = line['premise']
        question = line['question']
        if question == 'cause':
            question = 'What was the cause of this?'
        elif question == 'effect':
            question = 'What happened as a result?'
        endings = [line['choice{}'.format(i)] for i in range(1, 3)]
        if 'choice3' in line:
            endings += [line['choice3']]
        label = line['label']
        assert isinstance(question, str) and isinstance(contexts, str) and all([isinstance(e, str) for e in endings])
        examples.append(MCExample(guid=guid, question=question, contexts=contexts, endings=endings, label=label, language=language))
    return examples


def load_and_cache_examples(args, task, tokenizer, split='train', language='en',
                            prepare_trans=False, get_features=False):
    # Make sure only the first process in distributed training process the
    # dataset, and the others will use the cache
    if args.local_rank not in [-1, 0] and split != 'train':
        torch.distributed.barrier()
    if args.task_name in ['xcopa'] and split == 'dev':
        split = 'val'
    elif args.task_name in ['tydiqa']:
        if split == 'dev' and language != "en":
            split = 'train'
        elif split == 'test' and language != "en":
            split = "dev"

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_{}_{}".format(
            split,
            str(args.max_seq_length),
            str(task),
            str(language),
            tokenizer.name_or_path.replace("/", "_"),
            "for_nmt" if prepare_trans else "for_cls",
        ),
    )
    if os.path.exists(cached_features_file) and not get_features:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = task2labels[task]
        if task in ['xcopa']:
            examples = get_examples_MC(args.data_dir, language, split)
            convert_fn = convert_examples_MC_to_features
        elif task in ['tydiqa']:
            examples = TiDiQAProcessor().get_examples(args.data_dir, language, split)
        else:
            examples = get_examples_SC(args.data_dir, language, split)
            convert_fn = convert_examples_to_features

        if task in ['tydiqa']:
            features = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=(split == "train"),
                threads=1,
                )
        else:
            features = convert_fn(
                examples,
                tokenizer,
                label_list=label_list,
                max_length=args.max_seq_length,
                pad_on_left=False,
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=0,
                prepare_trans=prepare_trans,
            )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the
    # dataset, and the others will use the cache
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    if args.task_name not in ["tydiqa"]:
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long) if features[0].token_type_ids is not None else torch.zeros(all_input_ids.shape, dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    else:
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        if split != "train":
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_example_index,
            )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions
            )
    if get_features:
        return dataset, examples, features
    return dataset

def get_model(args, language, cls_tokenizer, model):
    if args.nmt_model_name == 'marian':
        nmt_model_name = f'Helsinki-NLP/opus-mt-{language}-en'
        nmt_model_list = [m.modelId for m in HfApi().model_list()]
        if nmt_model_name not in nmt_model_list:
            logger.warning("Marian NMT model %s-en does not exist!", language)
            return None, None

        nmt_model = MarianMTModel.from_pretrained(nmt_model_name)
        tokenizer = MarianTokenizer.from_pretrained(nmt_model_name, do_lower_case=False)
        forced_bos_token_id = None
    elif args.nmt_model_name == 'mbart50':
        nmt_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
        language2code = {lang.split("_")[0]: lang for lang in tokenizer.lang_code_to_id}
        if language not in language2code:
            logger.warning("MBART 50 model %s-en does not exist!", language)
            return None, None
        tokenizer.src_lang = language2code[language]
        forced_bos_token_id = tokenizer.lang_code_to_id['en_XX']
    elif args.nmt_model_name == 'm2m100':
        nmt_model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_1.2B')
        tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_1.2B')
        language2code = {lang.split("_")[0]: lang for lang in tokenizer.lang_code_to_id}
        if language not in language2code:
            logger.warning("M2M 100 model %s-en does not exist!", language)
            return None, None
        tokenizer.src_lang = language
        forced_bos_token_id = tokenizer.get_lang_id('en')

    if args.mode == 'trans-soft':
        def bind(instance, method):
            def binding_scope_fn(*args, **kwargs):
                return method(instance, *args, **kwargs)
            return binding_scope_fn
        nmt_model.generate_with_grad = bind(nmt_model, generate_with_grad)
    if args.weighted_ensemble:
        rr_config = RobertaConfig.from_pretrained("roberta-large", cache_dir=None)
        rr_config.num_sent = 2 if args.task_name == "xcopa" else 1
        reranker = RobertaForReranking.from_pretrained("roberta-large",
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=rr_config,
            cache_dir=None
            )
    else:
        reranker = None

    model = TranslatorClassifier(model, nmt_model, cls_tokenizer, tokenizer,
                                 args.num_samples, args.do_sample,
                                 args.max_seq_length, reranker,
                                 forced_bos_token_id=forced_bos_token_id,
                                 train_translator=(args.mode == 'trans-soft'),
                                 self_training=args.self_training)
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", required=True, choices=['pawsx', 'xnli', 'xcopa', 'tydiqa'],
                        help="""1) PAWS-X Paraphrase Identification.
                                2) XNLI Natural Language Inference.
                                3) XCOPA Commonsense Reasoning.
                                4) TyDIQA Question Answering.
                                """)
    parser.add_argument('--mode', type=str, required=True,
                        choices=['multi', 'trans-hard', 'trans-soft'],
                        help='Cross-lingual knowledge transfer method')
    parser.add_argument('--nmt_model_name', type=str, default="", choices=['', 'marian', 'google', 'mbart50', 'm2m100'],
                        help='pre-traslated data for hard translation')
    parser.add_argument('--cls_model_name', type=str, default="", choices=['xlmr', 'mbart50', 'm2m100', 'mt5'],
                        help='pre-traslated data for hard translation')
    parser.add_argument("--num_samples", default=1, type=int,
                        help="How many samples during generation.")
    parser.add_argument("--do_sample", action="store_true",
                        help="Whether to sample during generation.")
    parser.add_argument("--weighted_ensemble", action="store_true",
                        help="Whether to weight candidate translations by prob.")
    parser.add_argument("--self_training", action="store_true",
                        help="Whether to self-train the NMT system.")
    parser.add_argument("--output_dir", default="", type=str,
                        help="Output directory. If left unspecified, it equals `<task_name>_<mode>`")
    parser.add_argument("--model_dir", default="", type=str,
                        help="Model directory for fine-tuned classifiers. If left unspecified, it equals `<output_dir>/checkpoint-best`")
    parser.add_argument("--train_language", default="en", type=str,
                        help="Train language")
    parser.add_argument("--eval_languages", nargs='*',
                        help="Evaluation languages. All if empty.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="""The maximum total input sequence length after tokenization. Sequences longer
                             than this will be truncated, sequences shorter will be padded.""")
    parser.add_argument("--doc_stride", default=128, type=int)
    parser.add_argument("--max_query_length", default=64, type=int)
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation.")
    parser.add_argument("--do_refine", action="store_true", help="Whether to run refinement.")
    parser.add_argument("--do_analyse", action="store_true", help="Whether to run translation analysis.")
    parser.add_argument("--per_gpu_train_batch_size", default=6, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=12, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--log_file", default="train.log", type=str, help="log file")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA even when available")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="""For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].
                                See details at https://nvidia.github.io/apex/amp.html""")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
        )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
        )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
        )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
        )
    args = parser.parse_args()

    if args.mode == 'multi':
        assert not args.nmt_model_name and args.cls_model_name
    else:
        assert args.nmt_model_name and not args.cls_model_name
        if args.mode == 'trans-soft':
            assert args.nmt_model_name not in ["google"]
        elif args.mode == 'trans-hard' and args.nmt_model_name == 'google':
            assert not args.do_sample and not (args.num_samples > 1)

    data_extra_dir = 'google' if args.nmt_model_name == 'google' else ''
    args.data_dir = os.path.join('data', data_extra_dir, args.task_name)

    default_languages = set(task2eval_languages[args.task_name]) - set(mode2eval_languages[args.mode])
    if not args.eval_languages:
        args.eval_languages = sorted(list(default_languages))
    else:
        args.eval_languages = sorted(list(default_languages & set(args.eval_languages)))
    assert len(args.eval_languages) > 0

    if not args.output_dir:
        extra = [args.task_name, args.mode]
        if args.nmt_model_name:
            extra.append(args.nmt_model_name)
        if args.cls_model_name:
            extra.append(args.cls_model_name)
        if args.num_samples > 1:
            extra.append(str(args.num_samples))
        if args.do_sample:
            extra.append("sample")
        if args.weighted_ensemble:
            extra.append("reranked")
        if args.self_training:
            extra.append("selftrain")

        args.output_dir = os.path.join('results', "_".join(extra))

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(handlers=[logging.FileHandler(os.path.join(args.output_dir, args.log_file)), logging.StreamHandler()],
                        format='%(asctime)s - %(levelname)s - %(name)s -     %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logging.info("Input args: %r", args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which sychronizes nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    # Make sure only the first process in distributed training loads model & vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.mode in ["trans-soft", "trans-hard"]:
        config_class, tokenizer_class = (RobertaConfig, RobertaTokenizer)
        args.model_name_or_path = "roberta-large"
        if args.task_name in ["xcopa"]:
            model_class = RobertaForMultipleChoice
        elif args.task_name in ["tydiqa"]:
            model_class = RobertaForQuestionAnswering
        else:
            model_class = RobertaForSequenceClassification
    elif args.mode in ["multi"]:
        if args.cls_model_name == "xlmr":
            config_class, tokenizer_class = (XLMRobertaConfig, XLMRobertaTokenizer)
            args.model_name_or_path = "xlm-roberta-large"
            if args.task_name in ["xcopa"]:
                model_class = XLMRobertaForMultipleChoice
            elif args.task_name in ["tydiqa"]:
                model_class = XLMRobertaForQuestionAnswering
            else:
                model_class = XLMRobertaForSequenceClassification
        elif args.cls_model_name == "mbart50":
            config_class, tokenizer_class = (MBartConfig, MBart50TokenizerFast)
            args.model_name_or_path = "facebook/mbart-large-50"
            if args.task_name in ["xcopa"]:
                model_class = MBartForMultipleChoice
            elif args.task_name in ["tydiqa"]:
                model_class = MBartForQuestionAnswering
            else:
                model_class = MBartForSequenceClassification
        elif args.cls_model_name == "m2m100":
            config_class, tokenizer_class = (M2M100Config, M2M100Tokenizer)
            args.model_name_or_path = 'facebook/m2m100_1.2B'
            if args.task_name in ["xcopa"]:
                model_class = M2M100ForMultipleChoice
            else:
                raise NotImplementedError
        elif args.cls_model_name == "mt5":
            raise NotImplementedError

    config = config_class.from_pretrained(
        args.model_name_or_path,
        num_labels=len(task2labels[args.task_name]),
        finetuning_task=args.task_name,
        cache_dir=None,
    )
    logger.info("config = {}".format(config))

    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=False,
        cache_dir=None,
    )

    # Make sure only the first process in distributed training loads model & vocab
    if args.local_rank == 0:
        torch.distributed.barrier()
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        logger.info("loading from existing model {}".format(args.model_name_or_path))
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=None,
        )
        model.to(args.device)

        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, split='train',
                                                language=args.train_language)
        global_step, tr_loss, best_score, best_checkpoint = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        logger.info(" best checkpoint = {}, best score = {}".format(best_checkpoint, best_score))

    # Evaluation
    result = None
    best_checkpoint = os.path.join(args.output_dir, "checkpoint-best") if not args.model_dir else args.model_dir
    if args.do_eval and args.local_rank in [-1, 0]:
        output_predict_file = os.path.join(args.output_dir, 'results_zero.txt')
        total = total_correct = 0.0
        with open(output_predict_file, 'a') as writer:
            for language in args.eval_languages:
                tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=False)
                model = model_class.from_pretrained(best_checkpoint)

                prepare_trans = False
                if args.mode in ["trans-soft", "trans-hard"] and args.nmt_model_name != 'google' and language != "en":
                    cls_tokenizer = tokenizer
                    prepare_trans = True
                    tokenizer, model = get_model(args, language, cls_tokenizer, model)
                    if model is None:
                        continue

                model.to(args.device)
                result = evaluate(args, model, tokenizer, split='test', language=language,
                                  prefix='best_checkpoint', prepare_trans=prepare_trans)
                for key, value in result.items():
                    writer.write(f'{key}\t{language}\t{value}\n')
                    logger.info(f'{key}\t{language}\t{value}')

    # Refinement
    if args.do_refine:
        output_predict_file = os.path.join(args.output_dir, 'results_few.txt')
        total = total_correct = 0.0
        with open(output_predict_file, 'a') as writer:
            for language in args.eval_languages:
                if language == "en":
                    continue
                tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=False)
                model = model_class.from_pretrained(best_checkpoint)

                prepare_trans = False
                if args.mode in ["trans-soft", "trans-hard"] and args.nmt_model_name != 'google':
                    cls_tokenizer = tokenizer
                    prepare_trans = True
                    tokenizer, model = get_model(args, language, cls_tokenizer, model)
                    if model is None:
                        continue

                model.to(args.device)
                tgt_train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, split='dev',
                                                            language=language, prepare_trans=prepare_trans)
                global_step, tr_loss, result = refine(args, tgt_train_dataset, model, tokenizer, language, prepare_trans=prepare_trans)
                for key, value in result.items():
                    writer.write(f'{key}\t{language}\t{value}\n')
                    logger.info(f'{key}\t{language}\t{value}')

    # Analysis
    # TODO: add TyDiQA
    if args.do_analyse and args.local_rank in [-1, 0]:
        assert args.mode in ["trans-hard"] and args.nmt_model_name != 'google'
        translations_file = os.path.join(args.output_dir, 'translations.json')
        logits_matrix = {}
        with open(translations_file, 'w', encoding="utf-8") as writer:
            for language in args.eval_languages:
                if language == "en":
                    continue
                cls_tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=False)
                cls_model = model_class.from_pretrained(best_checkpoint)
                tokenizer, model = get_model(args, language, cls_tokenizer, cls_model)
                if model is None:
                    continue

                model.to(args.device)
                analyse(args, model, tokenizer, split='dev',
                        language=language, prefix='best_checkpoint',
                        writer=writer, logits_matrix=logits_matrix)
        logits_file = os.path.join(args.output_dir, 'logits_matrix.pkl')
        with open(logits_file, 'wb') as handle:
            pickle.dump(logits_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
