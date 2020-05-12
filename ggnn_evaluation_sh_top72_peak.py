import argparse
import yaml
import numpy as np
import torch
import pandas as pd
from lib import utils
from lib import metrics
from ggnn.multigraph import Net
from ggnn_train import _get_log_dir
from lib.utils import collate_wrapper

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def read_cfg_file(filename):
    with open(filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=Loader)
    return cfg


shtop72_index = np.array([
    5,
    19,
    86,
    82,
    152,
    89,
    195,
    198,
    168,
    214,
    24,
    18,
    94,
    187,
    21,
    84,
    189,
    61,
    14,
    217,
    88,
    22,
    148,
    91,
    9,
    169,
    17,
    64,
    270,
    190,
    60,
    197,
    6,
    167,
    11,
    240,
    202,
    20,
    146,
    45,
    40,
    193,
    47,
    62,
    158,
    196,
    63,
    191,
    46,
    28,
    44,
    199,
    87,
    35,
    68,
    30,
    33,
    41,
    59,
    2,
    37,
    4,
    29,
    10,
    0,
    34,
    39,
    38,
    36,
    7,
    15,
    12
])


def run_model(model, data_iterator, edge_index, edge_attr, device, output_dim):
    """
    return a list of (horizon_i, batch_size, num_nodes, output_dim)
    """
    # while evaluation, we need model.eval and torch.no_grad
    model.eval()
    y_pred_list = []
    for _, (x, y, xtime, ytime) in enumerate(data_iterator):
        y = y[..., :output_dim]
        sequences, y = collate_wrapper(x=x, y=y,
                                       edge_index=edge_index,
                                       edge_attr=edge_attr,
                                       device=device)
        # (T, N, num_nodes, num_out_channels)
        with torch.no_grad():
            y_pred = model(sequences)
            y_pred_list.append(y_pred.cpu().numpy())
    return y_pred_list


def evaluate(model,
             dataset,
             dataset_type,
             edge_index,
             edge_attr,
             device,
             output_dim,
             logger,
             detail=True,
             cfg=None,
             format_result=False):
    if detail:
        logger.info('Evaluation_{}_Begin:'.format(dataset_type))
    scaler = dataset['scaler']
    y_preds = run_model(
        model,
        data_iterator=dataset['{}_loader'.format(dataset_type)].get_iterator(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        device=device,
        output_dim=output_dim)

    y_preds = np.concatenate(y_preds, axis=0)  # concat in batch_size dim.
    mae_sum = 0
    mape_sum = 0
    rmse_sum = 0
    mae_list = []
    mape_list = []
    rmse_list = []
    mae_peak_list = []
    mape_peak_list = []
    rmse_peak_list = []
    # horizon = dataset['y_{}'.format(dataset_type)].shape[1]
    horizon = cfg['model']['horizon']
    ytime = dataset['ytime_{}'.format(dataset_type)][:, 0]
    ytime_pd = pd.to_datetime(ytime.flatten())
    morning_peak = np.logical_and(
        (ytime_pd.hour * 60 + ytime_pd.minute) >= 7 * 60 + 30,
        (ytime_pd.hour * 60 + ytime_pd.minute) <= 8 * 60 + 30)
    evening_peak = np.logical_and(
        (ytime_pd.hour * 60 + ytime_pd.minute) >= 17 * 60 + 30,
        (ytime_pd.hour * 60 + ytime_pd.minute) <= 18 * 60 + 30)
    peak = np.logical_or(morning_peak, evening_peak)
    for horizon_i in range(horizon):
        y_truth = scaler.inverse_transform(
            dataset['y_{}'.format(dataset_type)][:,
                                                 horizon_i,
                                                 shtop72_index, :output_dim])

        y_pred = scaler.inverse_transform(y_preds[:y_truth.shape[0],
                                                  horizon_i,
                                                  shtop72_index,
                                                  :output_dim])
        mae = metrics.masked_mae_np(y_pred, y_truth, null_val=0, mode='dcrnn')
        mape = metrics.masked_mape_np(y_pred, y_truth, null_val=0)
        rmse = metrics.masked_rmse_np(y_pred, y_truth, null_val=0)
        mae_sum += mae
        mape_sum += mape
        rmse_sum += rmse
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)

        # Peak
        y_truth_peak = scaler.inverse_transform(
            dataset['y_{}'.format(dataset_type)][peak,
                                                 horizon_i,
                                                 :, :output_dim])
        y_pred_peak = scaler.inverse_transform(y_preds[:y_truth.shape[0],
                                                       horizon_i, :,
                                                       :output_dim])
        y_pred_peak = y_pred_peak[peak]
        mae_peak = metrics.masked_mae_np(y_pred_peak,
                                         y_truth_peak,
                                         null_val=0,
                                         mode='dcrnn')
        mape_peak = metrics.masked_mape_np(y_pred_peak,
                                           y_truth_peak,
                                           null_val=0)
        rmse_peak = metrics.masked_rmse_np(y_pred_peak,
                                           y_truth_peak,
                                           null_val=0)
        mae_peak_list.append(mae_peak)
        mape_peak_list.append(mape_peak)
        rmse_peak_list.append(rmse_peak)

        msg = "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}"
        if detail:
            logger.info(msg.format(horizon_i + 1, mae, mape, rmse))
    if detail:
        logger.info('Evaluation_{}_End:'.format(dataset_type))
    if format_result:
        print('-' * 40 + ' 1/4 ' + '-' * 40)
        for i in range(len(mape_list)):
            print('{:.2f}'.format(mae_list[i]))
            print('{:.2f}%'.format(mape_list[i] * 100))
            print('{:.2f}'.format(rmse_list[i]))
            print()
        print('-' * 40 + ' Peak ' + '-' * 40)
        for i in range(len(mape_list)):
            print('{:.2f}'.format(mae_peak_list[i]))
            print('{:.2f}%'.format(mape_peak_list[i] * 100))
            print('{:.2f}'.format(rmse_peak_list[i]))
            print()
    else:
        return mae_sum / horizon, mape_sum / horizon, rmse_sum / horizon


def main(args):
    cfg = read_cfg_file(args.config_filename)
    log_dir = _get_log_dir(cfg)
    log_level = cfg.get('log_level', 'INFO')

    logger = utils.get_logger(log_dir, __name__, 'info.log', level=log_level)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    #  all edge_index in same dataset is same
    # edge_index = adjacency_to_edge_index(adj_mx)  # alreay added self-loop
    logger.info(cfg)
    # batch_size = cfg['data']['batch_size']
    # test_batch_size = cfg['data']['test_batch_size']
    # edge_index = utils.load_pickle(cfg['data']['edge_index_pkl_filename'])
    hz = cfg['data'].get('name', 'nothz') == 'hz'

    adj_mx_list = []
    graph_pkl_filename = cfg['data']['graph_pkl_filename']

    if not isinstance(graph_pkl_filename, list):
        graph_pkl_filename = [graph_pkl_filename]

    src = []
    dst = []
    for g in graph_pkl_filename:
        if hz:
            adj_mx = utils.load_graph_data_hz(g)
        else:
            _, _, adj_mx = utils.load_graph_data(g)

        for i in range(len(adj_mx)):
            adj_mx[i, i] = 0
        adj_mx_list.append(adj_mx)

    adj_mx = np.stack(adj_mx_list, axis=-1)
    if cfg['model'].get('norm', False):
        print('row normalization')
        adj_mx = adj_mx / (adj_mx.sum(axis=0) + 1e-18)
    src, dst = adj_mx.sum(axis=-1).nonzero()
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_attr = torch.tensor(adj_mx[adj_mx.sum(axis=-1) != 0],
                             dtype=torch.float,
                             device=device)

    output_dim = cfg['model']['output_dim']
    for i in range(adj_mx.shape[-1]):
        logger.info(adj_mx[..., i])

    #  print(adj_mx.shape) (207, 207)

    if hz:
        dataset = utils.load_dataset_hz(**cfg['data'],
                                        scaler_axis=(0,
                                                     1,
                                                     2,
                                                     3))
    else:
        dataset = utils.load_dataset(**cfg['data'])
    for k, v in dataset.items():
        if hasattr(v, 'shape'):
            logger.info((k, v.shape))

    model = Net(cfg).to(device)
    model.load_state_dict(torch.load(cfg['model']['save_path']), strict=False)

    evaluate(model=model,
             dataset=dataset,
             dataset_type='test',
             edge_index=edge_index,
             edge_attr=edge_attr,
             device=device,
             output_dim=output_dim,
             logger=logger,
             cfg=cfg,
             format_result=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename',
                        default=None,
                        type=str,
                        help='Configuration filename for restoring the model.')
    args = parser.parse_args()
    main(args)
