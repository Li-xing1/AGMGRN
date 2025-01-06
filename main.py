import json
import time
import torch.cuda
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils.Auxiliary import *
from utils.Metrics import *
from utils.Train import *
from datetime import datetime
import argparse


def get_paras():
    parser = argparse.ArgumentParser(description="Train and test a deep model for traffic forecasting.")
    parser.add_argument('--model_name', type=str, default='AGMGRN', help="traffic forecasting model")
    parser.add_argument('--dataset_name', type=str, default='HZMetro', help="traffic dataset")
    parser.add_argument('--epochs', type=int, default=1, help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--more_gpu', type=bool, default=True, help="Using multiple GPUs")
    parser.add_argument('--debug', type=bool, default=False, help="debug model")
    return parser.parse_args()


def train_epoch(train_loader, device, model, statics, criterion, optimizer, scheduler, args, norm_loss=True):
    ave = Average()

    statics = move2device(statics, device)
    for batch in tqdm(train_loader, unit='batch', desc='train'):
        inputs_norm, targets_norm, targets_unnorm, _, extras = batch
        inputs_norm, targets_norm, targets_unnorm, extras = move2device(
            [inputs_norm, targets_norm, targets_unnorm, extras], device)
        outputs = model(inputs_norm, targets_norm, extras, **statics)
        if norm_loss == True:
            loss = args['amp'] * criterion(outputs, targets_norm)
        else:
            outputs = denormalize(outputs, statics['nor_base'])
            loss = criterion(outputs, targets_unnorm)
        ave.add(loss.item(), targets_norm.shape[0])
        optimizer.zero_grad()
        loss.backward()
        if args['clip_grad_norm']:
            nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
        optimizer.step()
    scheduler.step()
    ave_loss = ave.average()

    return ave_loss


def val(val_loader, device, model, statics, args, mode, write):
    target, output, target_time_flag_list = [], [], []
    statics = move2device(statics, device)
    for batch in tqdm(val_loader, unit='batch', desc=mode):
        inputs_norm, _, targets_unnorm, target_time_flag, extras = batch
        inputs_norm, extras = move2device([inputs_norm, extras], device)
        with torch.no_grad():
            outputs = model(inputs_norm, None, extras, **statics)
        outputs = denormalize(outputs, statics['nor_base']).cpu()
        target.append(targets_unnorm)
        output.append(outputs)
        target_time_flag_list.append(target_time_flag)
    target, output, target_time_flag = torch.cat(target, dim=0), torch.cat(output, dim=0), torch.cat(
        target_time_flag_list,
        dim=0)
    if mode == 'test':
        tensorboard(write, target, output, out_len=args['datasets']['out_len'])
    rmse, mae, mape = Metrics(target, output, mode, mask=None).all()
    rmse_mask, mae_mask, mape_mask = Metrics(target, output, mode, mask=target_time_flag).all()
    return (rmse, mae, mape), (rmse_mask, mae_mask, mape_mask), output, target


def train(args, logger, write):
    device = args['device']

    # 数据库相关参数
    logger.info('--------- Dataset Info ---------')
    time_dataloader = time.time()
    (train_set, train_loader), (val_set, val_loader) = gen_train_val_data(args)
    statics = train_set.statics
    logger.info('dataset time: {:.6f}s'.format(time.time() - time_dataloader))

    try:
        statics, state_dict = load_model(osp.join(args['exp_dir'], 'best.pth'))
        model = build_model(args, mode='train', device=device, state_dict=state_dict)
    except:
        model = build_model(args, mode='train', device=device)
    model_sizes = model_size(model, type_size=4) / 1e6
    logger.info('--------- Model Info ---------')
    logger.info('Model size: {:.6f}MB'.format(model_sizes))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'],
                           eps=args['eps'], weight_decay=args['weight_decay'])

    # 学习率衰减
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[int(i * args['epochs']) for i in args['milestones']],
                                               gamma=args['gamma'])

    best_mae = np.inf
    ave_losses = []
    val_maes = []
    gpu_memory = 0.
    logger.info('---------- Training ----------')
    logger.info('num_samples: {}, num_batches: {}'.format(len(train_set), len(train_loader)))
    for epoch in range(1, args['epochs'] + 1):
        logger.info(f'-- Epoch:{epoch}/{args["epochs"]} --')
        start = time.time()
        ave_loss = train_epoch(train_loader, device, model, statics, criterion, optimizer, scheduler, args)
        time_elapsed = time.time() - start

        logger.info(
            f'[epoch {epoch}/{args["epochs"]}] ave_loss: {ave_loss:.6f}, time_elapsed: {time_elapsed:.6f}(sec)')
        ave_losses.append(ave_loss)
        write.add_scalar(tag='loss', scalar_value=ave_loss, global_step=epoch)
        memory = get_total_cache_usage()

        if memory > gpu_memory:
            gpu_memory = memory
        if not np.isnan(ave_loss):
            if (epoch) % args['val_freq'] == 0:
                logger.info('Validating...')
                logger.info('num_samples: {}, num_batches: {}'.format(len(val_set), len(val_loader)))

                model.eval()
                start = time.time()
                (_, mae, mape), _, _, _ = val(val_loader, device, model, statics, args, mode='val', write=write)
                time_elapsed = time.time() - start
                write.add_scalar(tag='val/mae', scalar_value=mae, global_step=epoch + 1)
                write.add_scalar(tag='val/mape', scalar_value=mape, global_step=epoch + 1)

                logger.info(f'time_elapsed: {time_elapsed:.6f}(sec)')

                if mae < best_mae:
                    best_mae = mae
                    best_epoch = epoch
                    # 保存最好情况下的模型
                    save_dict = {'model': model.state_dict(),
                                 'statics': statics,
                                 'epoch': epoch}
                    torch.save(save_dict, osp.join(args['exp_dir'], 'best.pth'))
                    logger.info("The best model 'best.pth' has been updated")
                else:
                    if epoch - best_epoch > args['patience_epochs']:
                        break
                if (epoch + 1) % args['save_every'] == 0:
                    save_dict = {'model': model.state_dict(),
                                 'statics': statics,
                                 'epoch': epoch}
                    torch.save(save_dict, osp.join(args['exp_dir'], 'epoch{:03d}.pth'.format(epoch)))
                    logger.info("The model 'epoch{:03d}.pth' has been saved".format(epoch))
                logger.info(
                    f'mae: {mae:.6f}, best_mae: {best_mae:.6f}，mape:{mape:.6f},best_epoch:{best_epoch}')
                val_maes.append([epoch, mae, mape])

                model.train()
        else:
            print('nan error')
            break
    np.savetxt(osp.join(args['exp_dir'], 'ave_losses.txt'), np.array(ave_losses))
    np.savetxt(osp.join(args['exp_dir'], 'val_maes.txt'), np.array(val_maes), '%d %g %g')
    return epoch, model_sizes, gpu_memory


def test(args, logger, write):
    device = args['device']

    test_set, test_loader = gen_test_data(args)
    statics, state_dict = load_model(osp.join(args['exp_dir'], 'best.pth'))
    model = build_model(args, mode='eval', device=device, state_dict=state_dict)

    logger.info('---------- Testing ----------')
    logger.info('num_samples: {}, num_batches: {}'.format(len(test_set), len(test_loader)))
    start = time.time()
    (rmse, mae, mape), (rmse_mask, mae_mask, mape_mask), output, target = val(test_loader, device, model, statics,
                                                                              args, mode='test', write=write)
    time_elapsed = time.time() - start
    infer_time = time_elapsed / len(test_loader)
    logger.info(f'time_elapsed: {time_elapsed:.6f}(sec)')
    metrics = save_metrics(rmse, mae, mape, args['exp_dir'], mask=False)
    metrics_mask = save_metrics(rmse_mask, mae_mask, mape_mask, args['exp_dir'], mask=False)
    logger.info(metrics)
    logger.info(metrics_mask)
    out = {'output': output, 'target': target}
    torch.save(out, osp.join(args['exp_dir'], 'output.pth'))

    return torch.mean(rmse).item(), torch.mean(mae).item(), torch.mean(mape).item(), infer_time


def train_test(args):
    args['exp_dir'] = create_exp_dir(args['dataset_name'], args['model_name'], args['exp_name'])
    logger = get_logger(args['exp_dir'])
    write = get_write(args)
    history = hisdict(args)

    if not args['test']:
        start_time = datetime.now()
        logger.info('Start time: {}'.format(start_time))
        logger.info('---------- Args ----------')
        logger.info(json.dumps(args, indent=2))

    if not args['test']:
        epoch, model_sizes, gpu_memory = train(args, logger, write)
        history['epoch'] = epoch
        history['model_sizes'] = model_sizes
        history['gpu_memory'] = gpu_memory

    rmse, mae, mape, infer_time = test(args, logger, write)
    history['infer_time'] = infer_time
    history['rmse'] = rmse
    history['mae'] = mae
    history['mape'] = mape
    logger.info('--------------------------')
    end_time = datetime.now()
    logger.info('End time: {}'.format(end_time))
    write.close()
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logging.shutdown()
    torch.cuda.empty_cache()

    if args['exp_name_mode'] == 'mape':
        new_name = f'exps/{args["dataset_name"]}/{args["model_name"]}/MAPE_{round(mape, 2)}%'
        old_name = f'exps/{args["dataset_name"]}/{args["model_name"]}/{args["exp_name"]}'
        os.rename(src=old_name, dst=new_name)
        if not args['test']:
            history['train_time'] = (end_time - start_time).seconds
            history['add'] = new_name
            append_df_to_excel(f'exps/{args["dataset_name"]}/{args["model_name"]}/history.xlsx', history)
            append_df_to_excel(f'exps/{args["dataset_name"]}/history.xlsx', history)
            args['test'] = True
            args['exp_name'] = f'MAPE_{round(mape, 2)}%'
        write2Yaml(args, new_name + f'/{args["dataset_name"]}_{args["model_name"]}.yaml')
    else:
        name = f'exps/{args["dataset_name"]}/{args["model_name"]}/{args["exp_name"]}'
        if not args['test']:
            history['time'] = (end_time - start_time).seconds
            history['add'] = name
            append_df_to_excel(f'exps/{args["dataset_name"]}/{args["model_name"]}/history.xlsx', history)
            append_df_to_excel(f'exps/{args["dataset_name"]}/history.xlsx', history)
            args['test'] = True
            args['exp_name'] = f'MAPE_{round(mape, 2)}%'
        write2Yaml(args, name + f'/{args["dataset_name"]}_{args["model_name"]}.yaml')


if __name__ == '__main__':
    paras = get_paras()
    args = conform_args(paras.model_name, paras.dataset_name, paras.epochs, paras.batch_size,
                        paras.more_gpu, paras.debug)
    # args = yaml.safe_load(open(add)) # when test
    train_test(args)
