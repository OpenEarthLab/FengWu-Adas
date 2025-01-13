import torch
import numpy as np

@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

@torch.jit.script
def latitude_weighting_factor_torch(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat)) / s

def weighted_latitude_weighting_factor_torch(j: torch.Tensor, real_num_lat:int, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return real_num_lat * torch.cos(3.1416/180. * lat(j, num_lat)) / s

# @torch.jit.script
def weighted_bias_torch_channels(pred: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t, num_lat, num_lat, s), (1, 1, -1, 1))

    result = torch.mean(weight * pred, dim=(-1,-2))

    # result = torch.sqrt(torch.mean(weight * (pred - torch.mean(weight * pred, dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
    return result

# @torch.jit.script
def weighted_bias_torch(pred: torch.Tensor) -> torch.Tensor:
    result = weighted_bias_torch_channels(pred)
    return torch.mean(result, dim=0)

def Bias(pred, gt, data_std):
    return weighted_bias_torch(pred - gt, metric_type="all") * data_std

@torch.jit.script
def weighted_rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
    return result

@torch.jit.script
def weighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_rmse_torch_channels(pred, target)
    return torch.mean(result, dim=0)

def WRMSE(pred, gt, data_std):
    return weighted_rmse_torch(pred, gt) * data_std

@torch.jit.script
def weighted_acc_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sum(weight * pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(weight * pred * pred, dim=(-1,-2)) * torch.sum(weight * target *
    target, dim=(-1,-2)))
    return result

@torch.jit.script
def weighted_acc_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_acc_torch_channels(pred, target)
    return torch.mean(result, dim=0)

def WACC(pred, gt, clim_time_mean_daily):
    return weighted_acc_torch(pred - clim_time_mean_daily, gt - clim_time_mean_daily)

class Metrics(object):
    """
    Define metrics for evaluation, metrics include:

        - MSE, masked MSE;

        - RMSE, masked RMSE;

        - REL, masked REL;

        - MAE, masked MAE;

        - Threshold, masked threshold.
    """
    def __init__(self, epsilon = 1e-8, **kwargs):
        """
        Initialization.

        Parameters
        ----------

        epsilon: float, optional, default: 1e-8, the epsilon used in the metric calculation.
        """
        super(Metrics, self).__init__()
        self.epsilon = epsilon
    
    def MSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        """
        MSE metric.

        Parameters
        ----------

        pred: tensor, required, the predicted;

        gt: tensor, required, the ground-truth

        Returns
        -------

        The MSE metric.
        """
        sample_mse = torch.mean((pred - gt) ** 2)
        return sample_mse.item()

    def RMSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        """
        RMSE metric.

        Parameters
        ----------

        pred: tensor, required, the predicted;

        gt: tensor, required, the ground-truth;


        Returns
        -------

        The RMSE metric.
        """
        sample_mse = torch.mean((pred - gt) ** 2, dim = [1, 2])
        return torch.mean(torch.sqrt(sample_mse)).item()
    
    def MAE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        """
        MAE metric.

        Parameters
        ----------
        pred: tensor, required, the predicted

        gt: tensor, required, the ground-truth

        Returns
        -------
        
        The MAE metric.
        """
        sample_mae = torch.mean(torch.abs(pred - gt))
        return sample_mae.item()

    def WRMSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        """
        WRMSE metric.

        Parameters
        ----------

        pred: tensor, required, the predicted;

        gt: tensor, required, the ground-truth;


        Returns
        -------

        The WRMSE metric.
        """

        return weighted_rmse_torch(pred, gt) * data_std

    def WACC(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        """
        WACC metric.

        Parameters
        ----------

        pred: tensor, required, the predicted;

        gt: tensor, required, the ground-truth;


        Returns
        -------

        The WACC metric.
        """

        return weighted_acc_torch(pred - clim_time_mean_daily, gt - clim_time_mean_daily)

class MetricsRecorder(object):
    """
    Metrics Recorder.
    """
    def __init__(self, metrics_list, epsilon = 1e-7, **kwargs):
        """
        Initialization.

        Parameters
        ----------

        metrics_list: list of str, required, the metrics name list used in the metric calcuation.

        epsilon: float, optional, default: 1e-8, the epsilon used in the metric calculation.
        """
        super(MetricsRecorder, self).__init__()
        self.epsilon = epsilon
        self.metrics = Metrics(epsilon = epsilon)
        self.metric_str_list = metrics_list
        self.metrics_list = []
        for metric in metrics_list:
            try:
                metric_func = getattr(self.metrics, metric)
                self.metrics_list.append([metric, metric_func, {}])
            except Exception:
                raise NotImplementedError('Invalid metric type.')
    
    def evaluate_batch(self, data_dict):
        """
        Evaluate a batch of the samples.

        Parameters
        ----------

        data_dict: pred and gt


        Returns
        -------

        The metrics dict.
        """
        pred = data_dict['pred']            # (B, C, H, W)
        gt = data_dict['gt']
        data_mask = None
        clim_time_mean_daily = None
        data_std = None
        if "clim_mean" in data_dict:
            clim_time_mean_daily = data_dict['clim_mean']    #(C, H, W)
            data_std = data_dict["std"]

        losses = {}
        for metric_line in self.metrics_list:
            metric_name, metric_func, metric_kwargs = metric_line
            loss = metric_func(pred, gt, data_mask, clim_time_mean_daily, data_std)
            if isinstance(loss, torch.Tensor):
                for i in range(len(loss)):
                    losses[metric_name+str(i)] = loss[i].item()
            else:
                losses[metric_name] = loss

        return losses

def _circumference(latitude):
    """Earth's circumference as a function of latitude."""
    circum_at_equator = 2 * np.pi * 1000 * (6357 + 6378) / 2

    result = torch.cos(latitude * torch.pi / 180) * circum_at_equator
    # print(result)
    return result

def lon_spacing_m(longitude, latitude):
    """Spacing (meters) between longitudinal values in `dataset`."""
    diffs = longitude.diff()
    return _circumference(latitude) * diffs[0] / 360

def spectrum_compute(data):
    """Computes zonal power at wavenumber and frequency."""
    _, _, num_latitude, num_longitude = data.shape
    latitude = torch.linspace(-90, 90, num_latitude).to(data)
    def simple_power(f_x):
        # print(f_x)
        f_k = torch.fft.rfft(f_x, dim=-1)/ num_longitude
        # print(f_k)
        # freq > 0 should be counted twice in power since it accounts for both
        # positive and negative complex values.
        one_and_many_twos = torch.cat((torch.tensor([1]), torch.tensor([2] * (f_k.shape[-1] - 1)))).to(f_x)
        result = torch.real(f_k * torch.conj(f_k)) * one_and_many_twos
        return result

    spectrum = simple_power(data)

    # This last step ensures the sum of spectral components is equal to the
    # (discrete) integral of data around a line of latitude.
    result = spectrum * _circumference(latitude).unsqueeze(-1)
    result = result.mean(dim=[0, -2])
    return result