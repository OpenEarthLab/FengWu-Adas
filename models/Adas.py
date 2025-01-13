import torch
import torch.nn as nn
import time
import json
import xarray as xr
import numpy as np
from timm.models.layers import trunc_normal_
import utils.misc as utils
from utils.metrics import WRMSE
from functools import partial
from .modules import AllPatchEmbed, PatchRecover, BasicLayer, SwinTransformerLayer
from utils.builder import get_optimizer, get_lr_scheduler
from datetime import datetime, timedelta


class Adas_model(nn.Module):
    def __init__(self, img_size=(69,721,1440), dim=192, patch_size=(1,6,6), window_size=(2,5,10), depth=8, num_heads=8,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, ape=True, use_checkpoint=True):
        super().__init__()

        self.patchembed = AllPatchEmbed(img_size=img_size, embed_dim=dim, patch_size=patch_size, norm_layer=nn.LayerNorm)  # b,c,14,180,360
        self.patchunembed = PatchRecover(img_size=img_size, embed_dim=dim, patch_size=patch_size)
        self.patch_resolution = self.patchembed.patch_resolution

        self.layer1 = BasicLayer(dim, kernel=(3,5,7), padding=(1,2,3), num_heads=num_heads, window_size=window_size, use_checkpoint=use_checkpoint)  # s1
        self.layer2 = BasicLayer(dim*2, kernel=(3,3,5), padding=(1,1,2), num_heads=num_heads, window_size=window_size, sample='down', use_checkpoint=use_checkpoint)  # s2
        self.layer3 = BasicLayer(dim*4, kernel=3, padding=1, num_heads=num_heads, window_size=window_size, sample='down', use_checkpoint=use_checkpoint)  # s3
        self.layer4 = BasicLayer(dim*2, kernel=(3,3,5), padding=(1,1,2), num_heads=num_heads, window_size=window_size, sample='up', use_checkpoint=use_checkpoint)  # s2
        self.layer5 = BasicLayer(dim, kernel=(3,5,7), padding=(1,2,3), num_heads=num_heads, window_size=window_size, sample='up', use_checkpoint=use_checkpoint)  # s1

        self.fusion = nn.Conv3d(dim*3, dim, kernel_size=(3,5,7), stride=1, padding=(1,2,3))

        # absolute position embedding
        self.ape = ape
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, self.patch_resolution[0], self.patch_resolution[1], self.patch_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.decoder = SwinTransformerLayer(dim=dim, depth=depth, num_heads=num_heads, window_size=window_size, qkv_bias=True, 
                                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, use_checkpoint=use_checkpoint)

        # initial weights
        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encoder_forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x = self.layer3(x2)
        x = self.layer4(x, x2)
        x = self.layer5(x, x1)

        return x

    def forward(self, background, observation, mask):

        x = self.patchembed(background, observation, mask)
        if self.ape:
            x = [ x[i] + self.absolute_pos_embed for i in range(3) ]

        x = self.encoder_forward(x)
        x = self.fusion(torch.cat(x, dim=1))
        x = self.decoder(x)

        x = self.patchunembed(x)
        return x
    

class Adas(object):
    
    def __init__(self, **model_params) -> None:
        super().__init__()

        params = model_params.get('params', {})
        criterion = model_params.get('criterion', 'UnifyMAE')
        self.optimizer_params = model_params.get('optimizer', {})
        self.scheduler_params = model_params.get('lr_scheduler', {})

        self.kernel = Adas_model(**params)
        self.best_loss = 999
        self.criterion = self.get_criterion(criterion)
        self.criterion_mae = nn.L1Loss()
        self.criterion_mse = nn.MSELoss()

        if utils.is_dist_avail_and_initialized():
            self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
            if self.device == torch.device('cpu'):
                raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def get_criterion(self, loss_type):
        if loss_type == 'UnifyMAE':
            return partial(self.unify_losses, criterion=nn.L1Loss())
        elif loss_type == 'UnifyMSE':
            return partial(self.unify_losses, criterion=nn.MSELoss())
        else:
            raise NotImplementedError('Invalid loss type.')

    def unify_losses(self, pred, target, criterion):
        loss_sum = 0
        unify_loss = criterion(pred[:,0,:,:], target[:,0,:,:])
        for i in range(1, len(pred[0])):
            loss = criterion(pred[:,i,:,:], target[:,i,:,:])
            loss_sum += loss / (loss/unify_loss).detach()
        return (loss_sum + unify_loss) / len(pred[0])
    
    def train(self, train_data_loader, valid_data_loader, logger, args):
        
        train_step = len(train_data_loader)
        valid_step = len(valid_data_loader)
        self.optimizer = get_optimizer(self.kernel, self.optimizer_params)
        self.scheduler = get_lr_scheduler(self.optimizer, self.scheduler_params, total_steps=train_step*args.max_epoch)

        for epoch in range(args.max_epoch):
            begin_time = time.time()
            self.kernel.train()
            
            for step, batch_data in enumerate(train_data_loader):
                # print(batch_data[1])
                batch_data = batch_data[0]

                if args.obs_type == 'simulation':
                    truth = batch_data[-1].to(self.device, non_blocking=True)
                    if step == 0:
                        inp_data = torch.cat([batch_data[0], batch_data[1]], dim=1).numpy()
                    mask = (torch.rand(truth.shape, device=self.device) >= args.mask_ratio).float()
                    background = args.forecast_model.run(None, {'input':inp_data})[0][:,:truth.shape[1]]
                    background = torch.from_numpy(background).to(self.device, non_blocking=True)
                    observation = truth * mask
                elif args.obs_type == 'gdas':
                    truth = batch_data[-3].to(self.device, non_blocking=True)
                    if step == 0:
                        inp_data = torch.cat([batch_data[0], batch_data[1]], dim=1).numpy()
                    background = args.forecast_model.run(None, {'input':inp_data})[0][:,:truth.shape[1]]
                    background = torch.from_numpy(background).to(self.device, non_blocking=True)
                    observation = batch_data[-2].to(self.device, non_blocking=True)
                    mask = batch_data[-1].to(self.device, non_blocking=True)
                else:
                    raise NotImplementedError('Invalid observation type.')

                self.optimizer.zero_grad()
                analysis = self.kernel(background, observation, mask)
                inp_data = np.concatenate([inp_data[:,truth.shape[1]:], analysis.detach().cpu().numpy()], axis=1)
                loss = self.criterion(analysis, truth)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                if ((step + 1) % 100 == 0) | (step+1 == train_step):
                    logger.info(f'Train epoch:[{epoch+1}/{args.max_epoch}], step:[{step+1}/{train_step}], lr:[{self.scheduler.get_last_lr()[0]}], loss:[{loss.item()}]')

            self.kernel.eval()
            with torch.no_grad():
                total_loss = 0

                for step, batch_data in enumerate(valid_data_loader):
                    # print(batch_data[1])
                    batch_data = batch_data[0]

                    if args.obs_type == 'simulation':
                        truth = batch_data[-1].to(self.device, non_blocking=True)
                        if step == 0:
                            inp_data = torch.cat([batch_data[0], batch_data[1]], dim=1).numpy()
                            mask = (torch.rand(truth.shape, device=self.device) >= args.mask_ratio).float()
                        background = args.forecast_model.run(None, {'input':inp_data})[0][:,:truth.shape[1]]
                        background = torch.from_numpy(background).to(self.device, non_blocking=True)
                        observation = truth * mask
                    elif args.obs_type == 'gdas':
                        truth = batch_data[-3].to(self.device, non_blocking=True)
                        if step == 0:
                            inp_data = torch.cat([batch_data[0], batch_data[1]], dim=1).numpy()
                        background = args.forecast_model.run(None, {'input':inp_data})[0][:,:truth.shape[1]]
                        background = torch.from_numpy(background).to(self.device, non_blocking=True)
                        observation = batch_data[-2].to(self.device, non_blocking=True)
                        mask = batch_data[-1].to(self.device, non_blocking=True)
                    else:
                        raise NotImplementedError('Invalid observation type.')

                    analysis = self.kernel(background, observation, mask)
                    inp_data = np.concatenate([inp_data[:,truth.shape[1]:], analysis.cpu().numpy()], axis=1)
                    loss = self.criterion(analysis, truth).item()
                    mae = self.criterion_mae(analysis, truth).item()
                    mse = self.criterion_mse(analysis, truth).item()
                    total_loss += loss

                    if ((step + 1) % 100 == 0) | (step+1 == valid_step):
                        logger.info(f'Valid epoch:[{epoch+1}/{args.max_epoch}], step:[{step+1}/{valid_step}], loss:[{loss}], MSE:[{mse}], MAE:[{mae}]')
        
            if (total_loss/valid_step) < self.best_loss:
                if utils.get_world_size() > 1 and utils.get_rank() == 0:
                    torch.save(self.kernel.module.state_dict(), f'{args.rundir}/best_model.pth')
                elif utils.get_world_size() == 1:
                    torch.save(self.kernel.state_dict(), f'{args.rundir}/best_model.pth')
                logger.info(f'New best model appears in epoch {epoch+1}.')
                self.best_loss = total_loss/valid_step
            logger.info(f'Epoch {epoch+1} average loss:[{total_loss/valid_step}], time:[{time.time()-begin_time}]')


    def test(self, test_data_loader, logger, args):

        if args.pred_len != 0:
            metric_logger = []
            for i in range(args.pred_len):
                metric_logger.append(utils.MetricLogger(delimiter="  "))

        if utils.get_world_size() > 1:
            rank = utils.get_rank()
            world_size = utils.get_world_size()
        else:
            rank = 0
            world_size = 1

        test_step = len(test_data_loader)
        data_mean, data_std = test_data_loader.dataset.get_meanstd()
        self.data_mean = data_mean.to(self.device)
        self.data_std = data_std.to(self.device)
        data_set_total_size = test_data_loader.sampler.total_size
        base_index = rank * (data_set_total_size // world_size)

        self.kernel.eval()
        with torch.no_grad():
            total_rmse_adas = []
            total_rmse_back = []
            total_pred_rmse = []

            for step, batch_data in enumerate(test_data_loader):
                date = batch_data[-1]
                current_date = datetime(int(date[0:4]), int(date[10:12]), int(date[13:15]), int(date[16:18]))
                batch_data = batch_data[0]

                if args.obs_type == 'simulation':
                    truth = batch_data[-1].to(self.device, non_blocking=True)
                    if step == 0:
                        inp_data = torch.cat([batch_data[0], batch_data[1]], dim=1).numpy()
                        mask = (torch.rand(truth.shape, device=self.device) >= args.mask_ratio).float()
                    background = args.forecast_model.run(None, {'input':inp_data})[0][:,:truth.shape[1]]
                    background = torch.from_numpy(background).to(self.device, non_blocking=True)
                    observation = truth * mask
                elif args.obs_type == 'gdas':
                    truth = batch_data[-3].to(self.device, non_blocking=True)
                    if step == 0:
                        inp_data = torch.cat([batch_data[0], batch_data[1]], dim=1).numpy()
                    background = args.forecast_model.run(None, {'input':inp_data})[0][:,:truth.shape[1]]
                    background = torch.from_numpy(background).to(self.device, non_blocking=True)
                    observation = batch_data[-2].to(self.device, non_blocking=True)
                    mask = batch_data[-1].to(self.device, non_blocking=True)
                else:
                    raise NotImplementedError('Invalid observation type.')
                
                analysis = self.kernel(background, observation, mask)
                inp_data = np.concatenate([inp_data[:,truth.shape[1]:], background.cpu().numpy()], axis=1)
                rmse_back = WRMSE(background, truth, self.data_std)
                rmse_adas = WRMSE(analysis, truth, self.data_std)
                total_rmse_adas.append(rmse_adas)
                total_rmse_back.append(rmse_back)

                logger.info("#"*80)
                logger.info(f'{step+1}/{test_step}: back_rmse = {rmse_back[11].item(), rmse_back[66].item()}, adas_rmse = {rmse_adas[11].item(), rmse_adas[66].item()}')
                
                if args.pred_len != 0 and args.eval_data == 'era5':
                    losses = self.multi_step_predict(inp_data, test_data_loader, step, base_index, args)
                    for i in range(len(losses)):
                        metric_logger[i].update(**losses[i])
                    if (step % 100 == 0) | (step + 1 == test_step):
                        for i in range(args.pred_len):
                            logger.info('  '.join(
                                [f'final test {i}th step predict',
                                "{meters}"]).format(
                                    meters=str(metric_logger[i])
                                ))
                            
                if args.pred_len != 0 and args.eval_data == 'igra' and (int(date[16:18]) == 0 or int(date[16:18]) == 12):
                    inp = inp_data
                    pred_all = []
                    for i in range(args.pred_len):
                        pred = args.forecast_model.run(None, {'input':inp})[0][:,:truth.shape[1]]
                        inp = np.concatenate([inp[:,truth.shape[1]:], pred], axis=1)
                        pred_all.append(pred)
                    pred_all = np.concatenate(pred_all, axis=0)
                    pred_all = torch.from_numpy(pred_all).to(self.device, non_blocking=True)

                    with open('./data/igra/igra_z500.json') as f:
                        igra_z500 = json.load(f)
                    with open('./data/igra/igra_t850.json') as f:
                        igra_t850 = json.load(f)
                    rmse_len = []
                    for i in range(args.pred_len // 2):
                        lead_date = current_date + timedelta(hours=12) * (i+1)
                        gt_z500 = np.array(igra_z500[lead_date.strftime("%Y%m%d%H")])
                        gt_t850 = np.array(igra_t850[lead_date.strftime("%Y%m%d%H")])
                        rmse_len.append(self.evaluate_on_igra(gt_z500, gt_t850, pred_all[2*i+1:2*i+2]))
                    total_pred_rmse.append(np.array(rmse_len))
                    logger.info(f'Pred RMSE at {date}: {rmse_len}')

            total_rmse_adas = torch.stack(total_rmse_adas, dim=0)
            total_rmse_back = torch.stack(total_rmse_back, dim=0)
            if total_pred_rmse:
                total_pred_rmse = np.mean(np.stack(total_pred_rmse, axis=0), axis=0)
                logger.info(f'Average Pred RMSE: {total_pred_rmse}')


    def evaluate_on_igra(self, gt_z500, gt_t850, pred_grid):
        grid_lat = np.load('./data/igra/latitude.npy')
        grid_lon = np.load('./data/igra/longitude.npy')
        pred_z500, pred_t850 = pred_grid[0,11], pred_grid[0,66]  # H, W
        pred_z500_grid = xr.DataArray((pred_z500 * self.data_std[11] + self.data_mean[11]).cpu().numpy(), \
                                      coords=[grid_lat, grid_lon], dims=['lat', 'lon'])
        pred_t850_grid = xr.DataArray((pred_t850 * self.data_std[66] + self.data_mean[66]).cpu().numpy(), \
                                      coords=[grid_lat, grid_lon], dims=['lat', 'lon'])
        pred_z500_station = pred_z500_grid.interp(coords={
            'lat': xr.DataArray(gt_z500[:,0], dims='z'),
            'lon': xr.DataArray(gt_z500[:,1], dims='z')}).values
        pred_t850_station = pred_t850_grid.interp(coords={
            'lat': xr.DataArray(gt_t850[:,0], dims='z'),
            'lon': xr.DataArray(gt_t850[:,1], dims='z')}).values
        rmse_z500 = np.sqrt(((gt_z500[:,2] - pred_z500_station) ** 2).sum() / len(gt_z500))
        rmse_t850 = np.sqrt(((gt_t850[:,2] - pred_t850_station) ** 2).sum() / len(gt_t850))
        return [rmse_z500, rmse_t850]


    def multi_step_predict(self, inp_data, test_data_loader, step, base_index, args):

        batch_len = inp_data.shape[0]
        sample_stride = test_data_loader.dataset.sample_stride
        file_stride = test_data_loader.dataset.file_stride
        inference_stride = test_data_loader.dataset.inference_stride
        use_gt = test_data_loader.dataset.use_gt
        index = ((step * batch_len) + base_index) * (inference_stride // sample_stride // file_stride) + 1

        metrics_losses = []
        for i in range(args.pred_len):
            if use_gt:
                tar_list = []
            clim_list = []
            for idx in range(index + i * sample_stride, index + i * sample_stride + batch_len):
                if use_gt:
                    # print(idx)
                    tar_list.append(torch.Tensor(test_data_loader.dataset.getitem(idx)[0][0]).float())
                    if "WACC" in args.metric_list:
                        clim_list.append(test_data_loader.dataset.get_clim_daily(idx))
            if use_gt:
                tar = torch.stack(tar_list, dim=0).to(self.device, non_blocking=True)
                if "WACC" in args.metric_list:
                    clim_data = torch.stack(clim_list, dim=0).to(self.device, non_blocking=True)
                else:
                    clim_data = None

            begin_time = time.time()
            pred = args.forecast_model.run(None, {'input':inp_data})[0][:,:tar.shape[1]]
            cost_time = time.time() - begin_time

            if use_gt:
                data_dict = {}
                data_dict['gt'] = tar
                data_dict['pred'] = torch.from_numpy(pred).to(self.device, non_blocking=True)
                data_dict['clim_mean'] = clim_data
                data_dict['std'] = self.data_std
                eval_loss_dict = args.eval_metrics.evaluate_batch(data_dict)
                eval_loss_dict["inference_time"] = cost_time
                metrics_losses.append(eval_loss_dict)
            else:
                metrics_losses.append({"inference_time": cost_time})
            inp_data = np.concatenate([inp_data[:,tar.shape[1]:], pred], axis=1)
            
        return metrics_losses