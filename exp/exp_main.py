import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from tqdm import tqdm
from models import CQMTformer
from models.Former import Autoformer,FEDformer,DRFormer,iTransformer,PatchTST
from models.CNN import MICN
from models.RNN import BiMamba4TS,S_Mamba
from models.MLP import DLinear,Linear,NLinear,TimeMixer,TimesNet,TimeXer

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'iTransformer': iTransformer,
            'PatchTST': PatchTST,
            'DRFormer': DRFormer,
            'BiMamba4TS': BiMamba4TS,
            'S_Mamba':S_Mamba,
            'DLinear': DLinear,
            'Linear': Linear,
            'NLinear': NLinear,
            'MICN': MICN,
            'TimeMixer': TimeMixer,
            'TimeXer': TimeXer,
            'TimesNet': TimesNet,
            'CQMTformer': CQMTformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _count_parameters(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    def _get_task_id_path(self, base_dir, setting):
        task_id = getattr(self.args, 'task_id', 'default_task')
        task_path = os.path.join(base_dir, f'task_{task_id}', setting)
        return task_path

    def _save_model_info(self, setting, model):
        total_params, trainable_params = self._count_parameters(model)

        info_dir = self._get_task_id_path('./model_info/', setting)
        if not os.path.exists(info_dir):
            os.makedirs(info_dir)

        info_file = os.path.join(info_dir, 'model_info.txt')
        with open(info_file, 'w') as f:
            f.write(f"Task ID: {getattr(self.args, 'task_id', 'default_task')}\n")
            f.write(f"Model Architecture: {self.args.model}\n")
            f.write(f"Setting: {setting}\n")
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n")
            f.write(f"Non-trainable Parameters: {total_params - trainable_params:,}\n")
            f.write(f"Parameter Size: {total_params * 4 / (1024 ** 2):.2f} MB (FP32)\n")
            f.write(f"Input Length: {self.args.seq_len}\n")
            f.write(f"Prediction Length: {self.args.pred_len}\n")
            f.write(f"Features: {self.args.features}\n")
            f.write(f"Enc Layers: {getattr(self.args, 'e_layers', 'N/A')}\n")
            f.write(f"Dec Layers: {getattr(self.args, 'd_layers', 'N/A')}\n")
            f.write(f"d_model: {getattr(self.args, 'd_model', 'N/A')}\n")
            f.write(f"d_ff: {getattr(self.args, 'd_ff', 'N/A')}\n")
            f.write(f"n_heads: {getattr(self.args, 'n_heads', 'N/A')}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        summary_dir = self._get_task_id_path('./model_info/', '')
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        summary_file = os.path.join(summary_dir, 'models_summary.txt')
        with open(summary_file, 'a') as f:
            f.write(
                f"{setting:<50} | {self.args.model:<15} | {total_params:>12,} | {trainable_params:>12,} | {total_params * 4 / (1024 ** 2):>8.2f} MB | {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        return total_params, trainable_params

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        val_pbar = tqdm(total=len(vali_loader), desc='Validating', leave=False)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss.item())

                val_pbar.update(1)

        val_pbar.close()
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        total_params, trainable_params = self._save_model_info(setting, self.model)

        print("\n" + "=" * 80)
        print("MODEL PARAMETER INFORMATION:")
        print(f"Task ID: {getattr(self.args, 'task_id', 'default_task')}")
        print(f"Model: {self.args.model}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Parameter Size: {total_params * 4 / (1024 ** 2):.2f} MB (FP32)")
        print("=" * 80)

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = self._get_task_id_path(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_losses = []
        val_losses = []
        test_losses = []

        epoch_pbar = tqdm(total=self.args.train_epochs, desc='Training Progress')

        start_time = time.time()

        print("\n" + "=" * 80)
        print(f"{'Epoch':^6} | {'Train Loss':^12} | {'Val Loss':^12} | {'Test Loss':^12} | {'Time':^8} | {'ETA':^10}")
        print("-" * 80)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            batch_pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}', leave=False)

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'lr': f'{model_optim.param_groups[0]["lr"]:.6f}'
                })
                batch_pbar.update(1)

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    batch_pbar.set_postfix({
                        'loss': f'{loss.item():.6f}',
                        'ETA': f'{left_time / 60:.1f}m'
                    })
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            batch_pbar.close()

            epoch_time_used = time.time() - epoch_time
            train_loss_avg = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            train_losses.append(train_loss_avg)
            val_losses.append(vali_loss)
            test_losses.append(test_loss)

            elapsed_time = time.time() - start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = self.args.train_epochs - epoch - 1
            total_remaining_time = avg_epoch_time * remaining_epochs

            epoch_pbar.set_postfix({
                'Train': f'{train_loss_avg:.4f}',
                'Val': f'{vali_loss:.4f}',
                'Test': f'{test_loss:.4f}',
                'ETA': f'{total_remaining_time / 60:.1f}m'
            })
            epoch_pbar.update(1)

            print(
                f"{epoch + 1:^6} | {train_loss_avg:^12.6f} | {vali_loss:^12.6f} | {test_loss:^12.6f} | {epoch_time_used:^8.1f}s | {total_remaining_time / 60:^10.1f}m")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("-" * 80)
                print("Early stopping triggered!")
                print("-" * 80)
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        epoch_pbar.close()

        print("=" * 80)
        print("Training Summary:")
        print(f"Task ID: {getattr(self.args, 'task_id', 'default_task')}")
        print(f"Best validation loss: {early_stopping.best_score:.6f}")
        print(f"Final training loss: {train_losses[-1]:.6f}")
        print(f"Final test loss: {test_losses[-1]:.6f}")
        print("=" * 80)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print(f"Loading model from checkpoint for testing...")
            model_path = self._get_task_id_path(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint.pth')))

            total_params, trainable_params = self._count_parameters(self.model)
            print(f"Loaded model parameters: {total_params:,} total, {trainable_params:,} trainable")

        preds = []
        trues = []
        
        # Save test results using a path that includes task_id
        folder_path = self._get_task_id_path('./test_results/', setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            test_pbar = tqdm(total=len(test_loader), desc='Testing')
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                # Update test progress bar
                test_pbar.update(1)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

            test_pbar.close()

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save (use a path that includes task_id)
        folder_path = self._get_task_id_path('./results/', setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        print("\n" + "=" * 50)
        print("Test Results:")
        print(f"Task ID: {getattr(self.args, 'task_id', 'default_task')}")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAPE: {mape:.6f}")
        print(f"MSPE: {mspe:.6f}")
        print("=" * 50)

        result_dir = self._get_task_id_path('./', '')
        result_file = os.path.join(result_dir, 'result.txt')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            
        with open(result_file, 'a') as f:
            f.write(f"Task ID: {getattr(self.args, 'task_id', 'default_task')} - {setting}  \n")
            f.write(f'mse:{mse:.6f}, mae:{mae:.6f}, rmse:{rmse:.6f}, mape:{mape:.6f}, mspe:{mspe:.6f}')
            f.write('\n')
            f.write('\n')

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:

            path = self._get_task_id_path(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            pred_pbar = tqdm(total=len(pred_loader), desc='Predicting')
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

                pred_pbar.update(1)

            pred_pbar.close()

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = self._get_task_id_path('./results/', setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return