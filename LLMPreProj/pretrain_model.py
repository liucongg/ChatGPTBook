# -*- coding: utf-8 -*- 
# @Time : 2023/4/11 11:38 
# @Author : JunkRoy 
# @E-mail: shenroy92@gmail.com
# @File : pretrain_model.py
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
from dataset.blocklm_utils import CollectedDataset
from model import GLMModel, glm_get_params_for_weight_decay_optimization
from dataset.tokenization import make_tokenizer
from model import PyTorchDistributedDataParallel, DistributedDataParallel
import torch
import mpu
from utils import print_rank_0, save_ds_checkpoint, get_checkpoint_name, ensure_directory_exists, \
    get_checkpoint_tracker_filename, report_memory
from utils import Timers
from configs.configs import get_config
from model.learning_rates import AnnealingLR
import random
import numpy as np
import math
import deepspeed
from dataset.dataset import PretrainDataset
from torch.utils.data import SequentialSampler, BatchSampler, DataLoader


def get_model(args):
    """
    初始化模型参数
    :param args: 配置项
    :return:
    """
    model = GLMModel(num_layers=args.num_layers,
                     vocab_size=args.vocab_size,
                     hidden_size=args.hidden_size,
                     num_attention_heads=args.num_attention_heads,
                     embedding_dropout_prob=args.hidden_dropout,
                     attention_dropout_prob=args.attention_dropout,
                     output_dropout_prob=args.hidden_dropout,
                     max_sequence_length=args.max_position_embeddings,
                     max_memory_length=args.mem_length,
                     checkpoint_activations=args.checkpoint_activations,
                     checkpoint_num_layers=args.checkpoint_num_layers,
                     parallel_output=False,
                     relative_encoding=args.transformer_xl,
                     block_position_encoding=args.block_lm and not args.masked_lm,
                     output_predict=True,
                     spell_length=None,
                     spell_func=args.prompt_func,
                     attention_scale=args.attention_scale)

    if args.freeze_transformer:
        model.freeze_transformer(tune_prefix_layers=args.tune_prefix_layers)
    return model


def get_tokenizer(args):
    add_sentinel_token = args.max_position_embeddings
    tokenizer = make_tokenizer(args.tokenizer_type, None, args.tokenizer_path, args.vocab_size,
                               args.tokenizer_model_type, add_block_symbols=args.block_lm, cache_dir=args.cache_dir,
                               add_sentinel_token=add_sentinel_token, add_task_mask=args.task_mask,
                               add_decoder_mask=args.block_mask_prob > 0.0 or args.context_mask_ratio > 0.0,
                               fix_command_token=args.fix_command_token)
    return tokenizer


def set_deepspeed_activation_checkpointing(args):
    deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    args.master_ip = os.getenv('MASTER_ADDR', 'localhost')
    args.master_port = os.getenv('MASTER_PORT', '6000')
    init_method += args.master_ip + ':' + args.master_port
    if hasattr(deepspeed, "init_distributed"):
        deepspeed.init_distributed(dist_backend=args.distributed_backend)
    else:
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size, rank=args.rank,
            init_method=init_method)

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Optional DeepSpeed Activation Checkpointing Features
    #
    if hasattr(args, "deepspeed") and args.deepspeed and args.deepspeed_activation_checkpointing:
        set_deepspeed_activation_checkpointing(args)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        mpu.model_parallel_cuda_manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def report_evaluate_metrics(summary_writer, prefix, loss, ppl, gpt_loss, bert_loss, sent_loss, multi_loss, step):
    string = ' validation loss at {}'.format(prefix)
    string += ' | LM loss: {:.6E}'.format(loss)
    string += ' | LM PPL: {:.6E}'.format(ppl)
    if gpt_loss != 0:
        string += ' | GPT loss: {:.6E}'.format(gpt_loss)
    if bert_loss != 0:
        string += ' | BERT loss: {:.6E}'.format(bert_loss)
    if sent_loss != 0:
        string += ' | Sent loss: {:.6E}'.format(sent_loss)
    if multi_loss != 0:
        string += ' | Multi loss: {:.6E}'.format(multi_loss)
    length = len(string) + 1
    print_rank_0('-' * 100)
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/valid_ppl', ppl, step)
        summary_writer.add_scalar(f'Train/valid_loss', loss, step)
        if gpt_loss != 0:
            summary_writer.add_scalar(f'Train/valid_gpt_loss', gpt_loss, step)
        if bert_loss != 0:
            summary_writer.add_scalar(f'Train/valid_bert_loss', bert_loss, step)
        if sent_loss != 0:
            summary_writer.add_scalar(f'Train/valid_sent_loss', sent_loss, step)
        if multi_loss != 0:
            summary_writer.add_scalar(f'Train/valid_multi_loss', multi_loss, step)


def get_batch(data, args):
    # 获取输入数据的相关要素
    keys = ['input_ids', 'loss_mask']
    if args.transformer_xl or args.block_lm:
        keys += ['target', 'attention_mask']
    if args.block_lm:
        keys += ['position_id']
    datatype = torch.int64

    # Broadcast 数据
    data_b = mpu.broadcast_data(keys, data, datatype)
    tokens = data_b['input_ids'].long()
    labels = data_b['target'].long()
    attention_mask = data_b['attention_mask'].long()
    loss_mask = data_b['loss_mask'].float()
    position_ids = data_b['position_id'].long()
    return tokens, labels, loss_mask, attention_mask, position_ids


tokenizer = None


def report_iteration_metrics(summary_writer, optimizer, lr, loss, elapsed_time, step, total_step, args):
    log_string = ' iteration {:8d}/{:8d} |'.format(step, total_step)
    log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(elapsed_time)
    log_string += ' learning rate {:.3E} |'.format(lr)
    log_string += ' lm loss {:.6E} |'.format(loss)
    if args.fp16:
        log_string += ' loss scale {:.1f} |'.format(
            optimizer.cur_scale if args.deepspeed else optimizer.loss_scale)
    print_rank_0(log_string)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/lr', lr, step)
        summary_writer.add_scalar(f'Train/train_loss', loss, step)
        summary_writer.add_scalar(f'Train/elapsed_time', elapsed_time, step)


def backward_step(optimizer, model, lm_loss, args, timers):
    """
    反向传播主函数，利用优化器更新模型参数
    :param optimizer:
    :param model:
    :param lm_loss:
    :param args:
    :param timers:
    :return:
    """

    # 当前的总loss
    loss = lm_loss

    # 反向传播
    if args.deepspeed:
        model.backward(loss)
    else:
        if args.fp16:
            optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

    timers('allreduce').reset()

    # 利用deepspeed更新参数
    if not args.deepspeed:
        if args.fp16:
            optimizer.update_master_grads()

        # Clipping gradients helps prevent the exploding gradient.
        if args.clip_grad > 0:
            if not args.fp16:
                mpu.clip_grad_norm(model.parameters(), args.clip_grad)
            else:
                optimizer.clip_master_grads(args.clip_grad)

    return lm_loss


def train_step(data_iterator, model, optimizer, lr_scheduler, args, timers, forward_step_func, mems=None,
               single_step=False):
    """
    单步训练函数，执行数据，返回loss
    :param data_iterator: 数据迭代器
    :param model: 模型
    :param optimizer: 优化器
    :param lr_scheduler: 调度器
    :param args: 配置参数
    :param timers:
    :param forward_step_func: 前向传播函数
    :param mems:
    :param single_step:
    :return:
    """
    lm_loss_total, count = 0.0, 0
    mems = [] if mems is None else mems
    if not args.deepspeed:
        optimizer.zero_grad()
    while True:
        skipped_iter, complete = 0, False
        # 执行前向传播，获取模型训练的loss
        timers('forward').start()
        lm_loss, mems, _ = forward_step_func(data_iterator, model, args, timers, mems)
        timers('forward').stop()
        if not args.deepspeed:
            lm_loss /= args.gradient_accumulation_steps

        reduced_loss = lm_loss.detach().clone().view(1)
        torch.distributed.all_reduce(reduced_loss.data, group=mpu.get_data_parallel_group())
        reduced_loss.data = reduced_loss.data / (args.world_size / args.model_parallel_size)

        lm_loss_total += reduced_loss
        count += 1

        # 计算梯度，并进行反向传播
        timers('backward').start()

        # 反向传播函数
        backward_step(optimizer, model, lm_loss, args, timers)
        timers('backward').stop()

        # 更新参数
        timers('optimizer').start()
        if args.deepspeed:
            if model.is_gradient_accumulation_boundary():
                model.step()
                complete = True
                if not (args.fp16 and optimizer.overflow):
                    lr_scheduler.step()
                else:
                    skipped_iter = 1
            else:
                model.step()
        else:
            if count == args.gradient_accumulation_steps:
                optimizer.step()
                complete = True
                # 更新学习率
                if not (args.fp16 and optimizer.overflow):
                    lr_scheduler.step()
                else:
                    skipped_iter = 1
        timers('optimizer').stop()
        if complete:
            break
        else:
            print_rank_0("Found NaN loss, skip backward")
            del lm_loss, reduced_loss
            mems = []
        if single_step:
            break
    if args.deepspeed:
        lm_loss_total = lm_loss_total / count
    return lm_loss_total, skipped_iter, mems


def forward_step(data_iterator, model, args, timers, mems):
    """
    前向传播主代码
    :param data_iterator: 数据迭代器
    :param model: 模型参数
    :param args: 配置参数
    :param timers:
    :param mems:
    :return:
    """

    # 获取batch
    timers('batch generator').start()
    timers('data loader').start()
    data = next(data_iterator[0]) if data_iterator[0] else None
    timers('data loader').stop()
    # 获取batch，必要的token、labels等参数
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data, args)
    timers('batch generator').stop()

    if data is not None and "mode" in data:
        mode = data['mode']
    else:
        mode = 'bert'
    # 前向传播模型，获取模型返回内容
    logits, *mems = model(tokens, position_ids, attention_mask, *mems)
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(),
                                              labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask)
    if loss_mask.sum().item() > 0:
        loss = loss / loss_mask.sum()

    return loss, mems, mode


def save_checkpoint(iteration, model, optimizer, lr_scheduler, args, tag=None, barrier=True,
                    only_changed_parameters=False, no_deepspeed=False, no_save_optim=False):
    """Save a model checkpoint."""
    if tag is None:
        tag = str(iteration)
    if args.deepspeed and not no_deepspeed:
        save_ds_checkpoint(iteration, model, lr_scheduler, args, tag=tag)
    else:
        # Only rank zer0 of the data parallel writes to the disk.

        if mpu.get_data_parallel_rank() == 0:
            checkpoint_name = get_checkpoint_name(args.save, tag)
            print('global rank {} is saving checkpoint at iteration {:7d} to {}'.
                  format(torch.distributed.get_rank(), iteration, checkpoint_name))
            sd = {'iteration': iteration}
            if args.deepspeed:
                model = model.module
            state_dict = model.state_dict()
            if only_changed_parameters:
                requires_grad_dict = {}
                for name, parameter in model.named_parameters():
                    requires_grad_dict[name] = parameter.requires_grad
                state_dict = {key: value for key, value in state_dict.items() if requires_grad_dict[key]}
            sd['module'] = state_dict

            # Optimizer stuff.
            if not args.no_save_optim and not no_save_optim:
                if optimizer is not None:
                    sd['optimizer'] = optimizer.state_dict()
                if lr_scheduler is not None:
                    sd['lr_scheduler'] = lr_scheduler.state_dict()

            # rng states.
            if not args.no_save_rng:
                sd['random_rng_state'] = random.getstate()
                sd['np_rng_state'] = np.random.get_state()
                sd['torch_rng_state'] = torch.get_rng_state()
                sd['cuda_rng_state'] = torch.cuda.get_rng_state()
                sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()

            ensure_directory_exists(checkpoint_name)
            torch.save(sd, checkpoint_name)
            print('  successfully saved {}'.format(checkpoint_name))

    # Wait so everyone is done (necessary)
    if barrier:
        torch.distributed.barrier()
    # And update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(tag)


def get_optimizer_param_groups(model):
    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (PyTorchDistributedDataParallel, DistributedDataParallel)):
        model = model.module
    param_groups = glm_get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        # print('## param_group', len(param_group['params']))
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    return param_groups


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    if args.finetune:
        num_iters = num_iters // args.gradient_accumulation_steps
    num_iters = max(1, num_iters)
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters - warmup_iter,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               decay_ratio=args.lr_decay_ratio)

    return lr_scheduler


def setup_model_and_optimizer(args, model_type=None, multi_token=True, num_labels=None, spell_length=None):
    """Setup model and optimizer."""

    model = get_model(args)
    param_groups = get_optimizer_param_groups(model)

    if args.train_data is not None or args.data_dir is not None and (args.epochs > 0 or args.train_iters > 0):
        print_rank_0("DeepSpeed is enabled.")

        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=param_groups,
            args=args,
            mpu=mpu,
            dist_init_required=False
        )
        lr_scheduler = get_learning_rate_scheduler(optimizer, args)
    else:
        optimizer, lr_scheduler = None, None

    return model, optimizer, lr_scheduler


def train(model, optimizer, lr_scheduler, train_data_iterator, val_data_iterator, timers, args, summary_writer=None):
    """
    训练模型的主函数

    :param model: 初始化的模型
    :param optimizer: 优化器
    :param lr_scheduler: 调度器
    :param train_data_iterator: 训练数据Dataloader
    :param val_data_iterator: 验证集Dataloader
    :param timers: 计时函数
    :param args: 配置信息
    :param summary_writer:
    :return:
    """

    # 将模型下置为训练模式，确保dropout可用
    model.train()

    # 设置loss为0
    total_lm_loss = 0.0

    # 迭代计数置为0
    skipped_iters = 0

    timers('interval time').start()
    report_memory_flag = True
    mems = []

    # 判断模型迭代
    while args.iteration < args.train_iters:
        # 将参数传入train_step函数，执行单步训练
        lm_loss, skipped_iter, mems = train_step(train_data_iterator, model, optimizer, lr_scheduler,
                                                 args, timers, mems=mems, forward_step_func=forward_step)
        skipped_iters += skipped_iter
        args.iteration += 1

        # 更新loss
        total_lm_loss += lm_loss.data.detach().float()

        # 更新日志信息
        if args.iteration % args.log_interval == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            elapsed_time = timers('interval time').elapsed()
            report_iteration_metrics(summary_writer, optimizer, learning_rate, avg_lm_loss,
                                     elapsed_time * 1000.0 / args.log_interval, args.iteration, args.train_iters, args)
            total_lm_loss = 0.0
            if report_memory_flag:
                report_memory('after {} iterations'.format(args.iteration))
                report_memory_flag = False
            if args.deepspeed or args.DDP_impl == 'torch':
                timers.log(['forward', 'backward', 'optimizer',
                            'batch generator', 'data loader'],
                           normalizer=args.log_interval)
            else:
                timers.log(['forward', 'backward', 'allreduce', 'optimizer',
                            'batch generator', 'data loader'],
                           normalizer=args.log_interval)

        # 根据save_interval，分阶段保存模型
        if args.save and args.save_interval and args.iteration % args.save_interval == 0:
            save_checkpoint(args.iteration, model, optimizer, lr_scheduler, args)

        # 验证模型效果
        if args.eval_interval and args.iteration % args.eval_interval == 0 and args.do_valid:
            prefix = 'iteration {}'.format(args.iteration)
            evaluate_and_print_results(prefix, val_data_iterator, model, args, timers, verbose=False,
                                       step=args.iteration, summary_writer=summary_writer,
                                       forward_step_func=forward_step)

    return args.iteration, skipped_iters


def evaluate(data_iterator, model, args, timers, forward_step_func, verbose=False):
    """
    模型推理代码
    :param data_iterator: 验证集数据迭代器
    :param model:  模型
    :param args: 配置参数
    :param timers: 时间信息
    :param forward_step_func: 前向传播函数
    :param verbose:
    :return:
    """
    # 将模型置为验证模式，确保关闭dropout
    model.eval()

    total_lm_loss, total_gpt_loss, total_bert_loss, total_sent_loss, total_multi_loss = 0, 0, 0, 0, 0
    gpt_iters, bert_iters, sent_iters, multi_iters = 0, 0, 0, 0
    mems = []
    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration, args.eval_iters))
            # 验证主逻辑.
            lm_loss, mems, mode = forward_step_func(data_iterator, model, args, timers, mems=mems)

            '''when contiguous memory optimizations are enabled, the buffers
            allocated by the optimizations are deallocated during backward pass
            in the absence of backward pass the buffers should be reset after each
            forward pass'''
            if args.deepspeed and args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

            lm_loss = lm_loss.data.detach().float().item()
            total_lm_loss += lm_loss
            if mode == 'gpt':
                total_gpt_loss += lm_loss
                gpt_iters += 1
            elif mode == 'bert':
                total_bert_loss += lm_loss
                bert_iters += 1
            elif mode == 'sentence':
                total_sent_loss += lm_loss
                sent_iters += 1
            elif mode == 'multi-task':
                total_multi_loss += lm_loss
                multi_iters += 1
    # 将模型重新置为训练模式.
    model.train()
    # Reduce across processes.
    loss_data = torch.cuda.FloatTensor(
        [total_lm_loss, total_gpt_loss, total_bert_loss, total_sent_loss, total_multi_loss, gpt_iters, bert_iters,
         sent_iters, multi_iters])
    torch.distributed.all_reduce(loss_data, group=mpu.get_data_parallel_group())
    loss_data = loss_data.tolist()
    total_lm_loss = loss_data[0] / args.eval_iters / (args.world_size / args.model_parallel_size)
    total_gpt_loss = loss_data[1] / loss_data[5] if loss_data[5] > 0 else 0
    total_bert_loss = loss_data[2] / loss_data[6] if loss_data[6] > 0 else 0
    total_sent_loss = loss_data[3] / loss_data[7] if loss_data[7] > 0 else 0
    total_multi_loss = loss_data[4] / loss_data[8] if loss_data[8] > 0 else 0
    return total_lm_loss, total_gpt_loss, total_bert_loss, total_sent_loss, total_multi_loss


def evaluate_and_print_results(prefix, data_iterator, model,
                               args, timers, forward_step_func, verbose=False, step=None, summary_writer=None):
    """Helper function to evaluate and dump results on screen."""
    lm_loss, gpt_loss, bert_loss, sent_loss, multi_loss = evaluate(data_iterator, model, args, timers, verbose=verbose,
                                                                   forward_step_func=forward_step_func)

    lm_ppl = math.exp(min(20, lm_loss))
    report_evaluate_metrics(summary_writer, prefix, lm_loss, lm_ppl, gpt_loss, bert_loss, sent_loss, multi_loss, step)

    return lm_loss


def main():
    """
    主训练函数，用于模型训练
    :return:
    """

    # 关闭 CuDNN.
    torch.backends.cudnn.enabled = False
    # 初始化时间类 Timer.
    timers = Timers()

    # 通过配置文件初始化相关配置.
    config_path = "./configs/configs.json"
    args = get_config(config_path)
    if args.load and not args.new_save_directory:
        args.experiment_name = os.path.basename(os.path.normpath(args.load))
    else:
        args.experiment_name = args.experiment_name
    if args.save:
        args.save = os.path.join(args.save, args.experiment_name)

    # 结合DeepSpeed对分布式训练相关参数进行初始化
    initialize_distributed(args)

    # 设置随机种子.
    set_random_seed(args.seed)

    # 初始化 tokenizer
    global tokenizer
    tokenizer = get_tokenizer(args)

    # 设置文件路径地址，初始化数据
    data_dir = "/data/work/syShen/models/glm_pretrain/data/"
    path_file = "/data/work/syShen/models/glm_pretrain/data/data.json"
    dataset = PretrainDataset(tokenizer, max_len=args.seq_length, data_dir=data_dir,
                              data_set_name="tmp", path_file=path_file, is_overwrite=False)
    # 初始化数据加载函数
    collate_fn = CollectedDataset(args, tokenizer, args.seq_length, bert_prob=args.bert_prob,
                                  gap_sentence_prob=args.gap_sentence_prob,
                                  gap_sentence_ratio=args.gap_sentence_ratio,
                                  gpt_infill_prob=args.gpt_infill_prob,
                                  average_block_length=args.avg_block_length,
                                  gpt_min_ratio=args.gpt_min_ratio,
                                  block_mask_prob=args.block_mask_prob,
                                  context_mask_ratio=args.context_mask_ratio,
                                  short_seq_prob=args.short_seq_prob,
                                  single_span_prob=args.single_span_prob,
                                  shuffle_blocks=not args.no_shuffle_block,
                                  block_position_encoding=not args.no_block_position,
                                  sentinel_token=args.sentinel_token,
                                  encoder_decoder=args.encoder_decoder,
                                  task_mask=args.task_mask, random_position=args.random_position,
                                  masked_lm=args.masked_lm).construct_blocks

    # 加载数据，结合collate_fn初始化Dataloader
    batch_sampler = BatchSampler(SequentialSampler(dataset), args.batch_size, drop_last=False)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers,
                             pin_memory=True, collate_fn=collate_fn)

    # 初始化模型、优化器、scheduler等必要参数
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    train_data_iterator = iter(data_loader)
    val_data_iterator = iter(data_loader)

    # 将计数器置为0
    args.iteration = 0

    # 将必要参数传入train函数中，开始模型训练
    train(model, optimizer, lr_scheduler, (train_data_iterator, None), (val_data_iterator, None), timers, args)


if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES=0,1  deepspeed pretrain_model.py