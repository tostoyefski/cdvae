from typing import Any, Dict  # 提供类型注解，增强代码可读性

import hydra  # 用于根据配置实例化对象
import numpy as np  # 进行数值计算
import omegaconf  # Hydra 的配置对象
import torch  # PyTorch 深度学习框架
import pytorch_lightning as pl  # 高级训练框架
import torch.nn as nn  # 神经网络模块
from torch.nn import functional as F  # 常用激活和损失函数
from torch_scatter import scatter  # 分布式张量聚合工具
from tqdm import tqdm  # 进度条库

from cdvae.common.utils import PROJECT_ROOT  # 项目根目录常量
from cdvae.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume,
    frac_to_cart_coords, min_distance_sqr_pbc)  # 数据处理工具函数
from cdvae.pl_modules.embeddings import MAX_ATOMIC_NUM  # 最大元素种类
from cdvae.pl_modules.embeddings import KHOT_EMBEDDINGS  # 元素嵌入向量


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    """构建一个简单的多层感知机"""  # 返回一个包含多层线性层的序列网络
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]  # 首层线性层和激活
    for i in range(fc_num_layers - 1):  # 根据配置堆叠隐藏层
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]  # 最后一层输出到目标维度
    return nn.Sequential(*mods)  # 组装为 nn.Sequential 对象


class BaseModule(pl.LightningModule):
    """LightningModule 的基类，封装通用的优化器配置"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()  # 初始化父类
        self.save_hyperparameters()  # 自动保存传入的参数到 hparams，方便日志记录

    def configure_optimizers(self):
        # 根据配置创建优化器
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:  # 若不使用学习率调度器
            return [opt]
        # 创建学习率调度器
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


class CrystGNN_Supervise(BaseModule):
    """
    GNN model for fitting the supervised objectives for crystals.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # 继承自 BaseModule

        self.encoder = hydra.utils.instantiate(self.hparams.encoder)  # 根据配置构建编码器

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        preds = self.encoder(batch)  # 直接调用编码器得到预测值，形状为 (N, 1)
        return preds

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch)  # 前向计算得到预测值

        loss = F.mse_loss(preds, batch.y)  # 监督任务使用均方误差
        self.log_dict(
            {'train_loss': loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch)  # 获取预测结果

        log_dict, loss = self.compute_stats(batch, preds, prefix='val')  # 计算评估指标

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch)  # 前向计算

        log_dict, loss = self.compute_stats(batch, preds, prefix='test')  # 统计测试指标

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, batch, preds, prefix):
        # 计算误差与评价指标
        loss = F.mse_loss(preds, batch.y)
        self.scaler.match_device(preds)  # 保证与模型设备一致
        scaled_preds = self.scaler.inverse_transform(preds)
        scaled_y = self.scaler.inverse_transform(batch.y)
        mae = torch.mean(torch.abs(scaled_preds - scaled_y))

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_mae': mae,
        }

        if self.hparams.data.prop == 'scaled_lattice':
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]
            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * \
                    batch.num_atoms.view(-1, 1).float()**(1/3)
            lengths_mae = torch.mean(torch.abs(pred_lengths - batch.lengths))
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mard = mard(batch.angles, pred_angles)

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(
                batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)
            log_dict.update({
                f'{prefix}_lengths_mae': lengths_mae,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mard': angles_mard,
                f'{prefix}_volumes_mard': volumes_mard,
            })
        return log_dict, loss


class CDVAE(BaseModule):
    """CDVAE 模型主体，包含编码器、解码器及各种损失计算"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # 调用父类初始化

        # 创建编码器和解码器
        self.encoder = hydra.utils.instantiate(
            self.hparams.encoder, num_targets=self.hparams.latent_dim)
        self.decoder = hydra.utils.instantiate(self.hparams.decoder)

        # VAE 中计算均值和方差的全连接层
        self.fc_mu = nn.Linear(self.hparams.latent_dim,
                               self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.latent_dim,
                                self.hparams.latent_dim)

        # 预测晶胞原子数、晶格以及组分的网络
        self.fc_num_atoms = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                      self.hparams.fc_num_layers, self.hparams.max_atoms+1)
        self.fc_lattice = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                    self.hparams.fc_num_layers, 6)
        self.fc_composition = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                        self.hparams.fc_num_layers, MAX_ATOMIC_NUM)
        # for property prediction.
        if self.hparams.predict_property:
            # 如果需要预测性质，额外构建一个 MLP
            self.fc_property = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                         self.hparams.fc_num_layers, 1)

        # 生成不同噪声水平的 sigma 值
        sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.sigma_begin),
            np.log(self.hparams.sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)  # 坐标噪声级别

        self.sigmas = nn.Parameter(sigmas, requires_grad=False)  # 不参与梯度更新

        type_sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.type_sigma_begin),
            np.log(self.hparams.type_sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)  # 原子类型噪声级别

        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)

        self.embedding = torch.zeros(100, 92)
        for i in range(100):
            # 预先存储一份元素的 one-hot 编码
            self.embedding[i] = torch.tensor(KHOT_EMBEDDINGS[i + 1])

        # obtain from datamodule.
        self.lattice_scaler = None  # 将在数据模块中填充
        self.scaler = None

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        # 这里实现重参数化技巧，从正态分布采样
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, batch):
        """
        encode crystal structures to latents.
        """
        # 先通过编码器得到隐藏表示
        hidden = self.encoder(batch)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z

    def decode_stats(self, z, gt_num_atoms=None, gt_lengths=None, gt_angles=None,
                     teacher_forcing=False):
        """
        decode key stats from latent embeddings.
        batch is input during training for teach-forcing.
        """
        # 根据是否使用教师强制来解码晶格相关信息
        if gt_num_atoms is not None:
            num_atoms = self.predict_num_atoms(z)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, gt_num_atoms))
            composition_per_atom = self.predict_composition(z, gt_num_atoms)
            if self.hparams.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            num_atoms = self.predict_num_atoms(z).argmax(dim=-1)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, num_atoms))
            composition_per_atom = self.predict_composition(z, num_atoms)
        # 返回预测结果
        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom

    @torch.no_grad()
    def langevin_dynamics(self, z, ld_kwargs, gt_num_atoms=None, gt_atom_types=None):
        """
        decode crystral structure from latent embeddings.
        ld_kwargs: args for doing annealed langevin dynamics sampling:
            n_step_each:  number of steps for each sigma level.
            step_lr:      step size param.
            min_sigma:    minimum sigma to use in annealed langevin dynamics.
            save_traj:    if <True>, save the entire LD trajectory.
            disable_bar:  disable the progress bar of langevin dynamics.
        gt_num_atoms: if not <None>, use the ground truth number of atoms.
        gt_atom_types: if not <None>, use the ground truth atom types.
        """
        # 如果需要保存轨迹，初始化缓存列表
        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []

        # 获取晶体的基本统计信息
        num_atoms, _, lengths, angles, composition_per_atom = self.decode_stats(
            z, gt_num_atoms)
        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms

        # 获取原子类型
        composition_per_atom = F.softmax(composition_per_atom, dim=-1)
        if gt_atom_types is None:
            cur_atom_types = self.sample_composition(
                composition_per_atom, num_atoms)
        else:
            cur_atom_types = gt_atom_types

        # 初始化分数坐标
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=z.device)

        # annealed langevin dynamics.
        # 逐个 sigma 进行退火采样
        for sigma in tqdm(self.sigmas, total=self.sigmas.size(0), disable=ld_kwargs.disable_bar):  # 遍历所有噪声水平
            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2

            for step in range(ld_kwargs.n_step_each):
                # 在坐标上加入噪声
                noise_cart = torch.randn_like(
                    cur_frac_coords) * torch.sqrt(step_size * 2)  # 按当前步长生成噪声
                # 解码器预测坐标差分和原子类型
                pred_cart_coord_diff, pred_atom_types = self.decoder(
                    z, cur_frac_coords, cur_atom_types, num_atoms, lengths, angles)  # 预测坐标和类型
                cur_cart_coords = frac_to_cart_coords(
                    cur_frac_coords, lengths, angles, num_atoms)  # 分数坐标转笛卡尔
                pred_cart_coord_diff = pred_cart_coord_diff / sigma  # 按噪声水平缩放梯度
                # 根据预测的差分和噪声更新坐标
                cur_cart_coords = cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart  # Euler 更新
                # 将笛卡尔坐标重新转换回分数坐标
                cur_frac_coords = cart_to_frac_coords(
                    cur_cart_coords, lengths, angles, num_atoms)  # 转回分数坐标

                if gt_atom_types is None:
                    # 根据预测结果更新原子类型
                    cur_atom_types = torch.argmax(pred_atom_types, dim=1) + 1

                if ld_kwargs.save_traj:
                    # 保存当前步的坐标与噪声等信息
                    all_frac_coords.append(cur_frac_coords)
                    all_pred_cart_coord_diff.append(
                        step_size * pred_cart_coord_diff)
                    all_noise_cart.append(noise_cart)
                    all_atom_types.append(cur_atom_types)

        # 最终返回的结构信息
        output_dict = {'num_atoms': num_atoms, 'lengths': lengths, 'angles': angles,
                       'frac_coords': cur_frac_coords, 'atom_types': cur_atom_types,
                       'is_traj': False}  # 最终采样得到的结构

        # 如果保存轨迹，将中间结果堆叠返回
        if ld_kwargs.save_traj:
            # 如果需要，附加整条采样轨迹
            output_dict.update(dict(
                all_frac_coords=torch.stack(all_frac_coords, dim=0),
                all_atom_types=torch.stack(all_atom_types, dim=0),
                all_pred_cart_coord_diff=torch.stack(
                    all_pred_cart_coord_diff, dim=0),
                all_noise_cart=torch.stack(all_noise_cart, dim=0),
                is_traj=True))

        return output_dict

    def sample(self, num_samples, ld_kwargs):
        # 从标准正态分布采样隐向量，然后通过 LD 生成结构
        z = torch.randn(num_samples, self.hparams.hidden_dim,
                        device=self.device)
        samples = self.langevin_dynamics(z, ld_kwargs)
        return samples

    def forward(self, batch, teacher_forcing, training):
        # 前向过程：编码 -> 解码 -> 计算损失
        # 这里加入了一些临时处理以避免 NaN，后续可进一步调试
        mu, log_var, z = self.encode(batch)

        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles,
         pred_composition_per_atom) = self.decode_stats(
            z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing)

        # 为每个晶体随机选择噪声级别
        noise_level = torch.randint(0, self.sigmas.size(0),
                                    (batch.num_atoms.size(0),),
                                    device=self.device)  # 为每个晶体随机选择噪声级别
        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(
            batch.num_atoms, dim=0)

        type_noise_level = torch.randint(0, self.type_sigmas.size(0),
                                         (batch.num_atoms.size(0),),
                                         device=self.device)  # 类型噪声级别
        used_type_sigmas_per_atom = (
            self.type_sigmas[type_noise_level].repeat_interleave(
                batch.num_atoms, dim=0))

        # 对原子类型添加噪声后重新采样
        pred_composition_probs = F.softmax(
            pred_composition_per_atom.detach(), dim=-1)  # 概率分布
        atom_type_probs = (
            F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM) +
            pred_composition_probs * used_type_sigmas_per_atom[:, None])
        rand_atom_types = torch.multinomial(
            atom_type_probs, num_samples=1).squeeze(1) + 1

        # 对坐标添加噪声
        cart_noises_per_atom = (
            torch.randn_like(batch.frac_coords) *
            used_sigmas_per_atom[:, None])  # 坐标噪声
        # 先转为笛卡尔坐标再加噪声，之后再转回分数坐标
        cart_coords = frac_to_cart_coords(
            batch.frac_coords, pred_lengths, pred_angles, batch.num_atoms)  # 转为笛卡尔坐标
        cart_coords = cart_coords + cart_noises_per_atom  # 加上噪声
        noisy_frac_coords = cart_to_frac_coords(
            cart_coords, pred_lengths, pred_angles, batch.num_atoms)  # 再转回分数坐标

        # 解码器根据当前坐标和原子类型预测修正量
        pred_cart_coord_diff, pred_atom_types = self.decoder(
            z, noisy_frac_coords, rand_atom_types, batch.num_atoms, pred_lengths, pred_angles)  # 解码预测

        # 计算各项损失
        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)  # 原子数损失
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)  # 晶格损失
        composition_loss = self.composition_loss(
            pred_composition_per_atom, batch.atom_types, batch)  # 组成损失
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch)  # 坐标损失
        type_loss = self.type_loss(pred_atom_types, batch.atom_types,
                                   used_type_sigmas_per_atom, batch)  # 类型损失

        kld_loss = self.kld_loss(mu, log_var)  # KL 散度损失

        if self.hparams.predict_property:
            property_loss = self.property_loss(z, batch)  # 额外性质预测损失
        else:
            property_loss = 0.

        # 返回所有预测和损失，用于后续统计
        # 返回所有预测与损失，供日志记录使用
        return {
            'num_atom_loss': num_atom_loss,
            'lattice_loss': lattice_loss,
            'composition_loss': composition_loss,
            'coord_loss': coord_loss,
            'type_loss': type_loss,
            'kld_loss': kld_loss,
            'property_loss': property_loss,
            'pred_num_atoms': pred_num_atoms,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': pred_cart_coord_diff,
            'pred_atom_types': pred_atom_types,
            'pred_composition_per_atom': pred_composition_per_atom,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': noisy_frac_coords,
            'rand_atom_types': rand_atom_types,
            'z': z,
        }

    def generate_rand_init(self, pred_composition_per_atom, pred_lengths,
                           pred_angles, num_atoms, batch):
        # 随机初始化坐标和原子类型，用于生成过程的起点
        rand_frac_coords = torch.rand(num_atoms.sum(), 3,
                                      device=num_atoms.device)  # 随机初始化坐标
        pred_composition_per_atom = F.softmax(pred_composition_per_atom,
                                              dim=-1)
        rand_atom_types = self.sample_composition(
            pred_composition_per_atom, num_atoms)
        return rand_frac_coords, rand_atom_types

    def sample_composition(self, composition_prob, num_atoms):
        """
        Samples composition such that it exactly satisfies composition_prob
        """
        # 根据给定的组成概率采样原子类型
        batch = torch.arange(
            len(num_atoms), device=num_atoms.device).repeat_interleave(num_atoms)  # 标记所属晶体
        assert composition_prob.size(0) == num_atoms.sum() == batch.size(0)
        composition_prob = scatter(
            composition_prob, index=batch, dim=0, reduce='mean')  # 计算晶体级别组成

        all_sampled_comp = []  # 存储每个晶体采样的组成

        for comp_prob, num_atom in zip(list(composition_prob), list(num_atoms)):
            # 根据概率计算每种元素应有的数量
            comp_num = torch.round(comp_prob * num_atom)
            atom_type = torch.nonzero(comp_num, as_tuple=True)[0] + 1
            atom_num = comp_num[atom_type - 1].long()

            # 将每种原子类型重复相应次数
            sampled_comp = atom_type.repeat_interleave(atom_num, dim=0)  # 按数量复制

            # 如果四舍五入后原子数不足，再随机补齐
            if sampled_comp.size(0) < num_atom:
                left_atom_num = num_atom - sampled_comp.size(0)

                left_comp_prob = comp_prob - comp_num.float() / num_atom  # 剩余的概率

                left_comp_prob[left_comp_prob < 0.] = 0.
                left_comp = torch.multinomial(
                    left_comp_prob, num_samples=left_atom_num, replacement=True)
                # convert to atomic number
                left_comp = left_comp + 1  # 转换为原子序数
                sampled_comp = torch.cat([sampled_comp, left_comp], dim=0)

            sampled_comp = sampled_comp[torch.randperm(sampled_comp.size(0))]  # 打乱顺序
            sampled_comp = sampled_comp[:num_atom]
            all_sampled_comp.append(sampled_comp)

        all_sampled_comp = torch.cat(all_sampled_comp, dim=0)
        assert all_sampled_comp.size(0) == num_atoms.sum()
        return all_sampled_comp

    def predict_num_atoms(self, z):
        # 预测每个晶体的原子数分布
        return self.fc_num_atoms(z)

    def predict_property(self, z):
        self.scaler.match_device(z)
        return self.scaler.inverse_transform(self.fc_property(z))

    def predict_lattice(self, z, num_atoms):
        # 预测晶格参数并进行反标准化
        self.lattice_scaler.match_device(z)
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        scaled_preds = self.lattice_scaler.inverse_transform(
            pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        if self.hparams.data.lattice_scale_method == 'scale_length':
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float()**(1/3)
        # <pred_lengths_and_angles> is scaled.
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def predict_composition(self, z, num_atoms):
        # 将晶体级别的隐向量扩展到原子级别后预测组成
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom

    def num_atom_loss(self, pred_num_atoms, batch):
        # 原子数的交叉熵损失
        return F.cross_entropy(pred_num_atoms, batch.num_atoms)

    def property_loss(self, z, batch):
        # 性质预测使用均方误差
        return F.mse_loss(self.fc_property(z), batch.y)

    def lattice_loss(self, pred_lengths_and_angles, batch):
        # 计算晶格参数的均方误差
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        if self.hparams.data.lattice_scale_method == 'scale_length':
            target_lengths = batch.lengths / \
                batch.num_atoms.view(-1, 1).float()**(1/3)
        target_lengths_and_angles = torch.cat(
            [target_lengths, batch.angles], dim=-1)
        target_lengths_and_angles = self.lattice_scaler.transform(
            target_lengths_and_angles)
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch):
        # 组成预测同样使用交叉熵
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom,
                               target_atom_types, reduction='none')
        return scatter(loss, batch.batch, reduce='mean').mean()

    def coord_loss(self, pred_cart_coord_diff, noisy_frac_coords,
                   used_sigmas_per_atom, batch):
        # 计算坐标预测的 MSE 损失
        noisy_cart_coords = frac_to_cart_coords(
            noisy_frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        target_cart_coords = frac_to_cart_coords(
            batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        # 根据 PBC 计算目标与噪声坐标之间的最小向量差
        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords, noisy_cart_coords, batch.lengths, batch.angles,
            batch.num_atoms, self.device, return_vector=True)

        # 根据噪声水平缩放误差和预测
        target_cart_coord_diff = target_cart_coord_diff / \
            used_sigmas_per_atom[:, None]**2
        pred_cart_coord_diff = pred_cart_coord_diff / \
            used_sigmas_per_atom[:, None]

        loss_per_atom = torch.sum(
            (target_cart_coord_diff - pred_cart_coord_diff)**2, dim=1)

        # 按照噪声大小重新加权
        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

    def type_loss(self, pred_atom_types, target_atom_types,
                  used_type_sigmas_per_atom, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(
            pred_atom_types, target_atom_types, reduction='none')
        # 根据噪声大小重新缩放损失
        loss = loss / used_type_sigmas_per_atom
        return scatter(loss, batch.batch, reduce='mean').mean()

    def kld_loss(self, mu, log_var):
        # Kullback-Leibler 散度
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        return kld_loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # 按照当前 epoch 决定是否使用教师强制
        teacher_forcing = (
            self.current_epoch <= self.hparams.teacher_forcing_max_epoch)
        outputs = self(batch, teacher_forcing, training=True)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, batch, outputs, prefix):
        # 各项损失值
        num_atom_loss = outputs['num_atom_loss']
        lattice_loss = outputs['lattice_loss']
        coord_loss = outputs['coord_loss']
        type_loss = outputs['type_loss']
        kld_loss = outputs['kld_loss']
        composition_loss = outputs['composition_loss']
        property_loss = outputs['property_loss']

        loss = (
            self.hparams.cost_natom * num_atom_loss +
            self.hparams.cost_lattice * lattice_loss +
            self.hparams.cost_coord * coord_loss +
            self.hparams.cost_type * type_loss +
            self.hparams.beta * kld_loss +
            self.hparams.cost_composition * composition_loss +
            self.hparams.cost_property * property_loss)  # 汇总总损失

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_natom_loss': num_atom_loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_coord_loss': coord_loss,
            f'{prefix}_type_loss': type_loss,
            f'{prefix}_kld_loss': kld_loss,
            f'{prefix}_composition_loss': composition_loss,
        }

        if prefix != 'train':
            # 验证和测试阶段的总损失仅考虑坐标和类型
            loss = (
                self.hparams.cost_coord * coord_loss +
                self.hparams.cost_type * type_loss)

            # evaluate num_atom prediction.
            pred_num_atoms = outputs['pred_num_atoms'].argmax(dim=-1)
            num_atom_accuracy = (
                pred_num_atoms == batch.num_atoms).sum() / batch.num_graphs  # 数量预测准确率

            # evalute lattice prediction.
            pred_lengths_and_angles = outputs['pred_lengths_and_angles']  # 晶格预测
            scaled_preds = self.lattice_scaler.inverse_transform(
                pred_lengths_and_angles)
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]

            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * \
                    batch.num_atoms.view(-1, 1).float()**(1/3)
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(
                batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)

            # evaluate atom type prediction.
            pred_atom_types = outputs['pred_atom_types']
            target_atom_types = outputs['target_atom_types']
            type_accuracy = pred_atom_types.argmax(
                dim=-1) == (target_atom_types - 1)  # 类型预测准确率
            type_accuracy = scatter(type_accuracy.float(
            ), batch.batch, dim=0, reduce='mean').mean()

            log_dict.update({
                f'{prefix}_loss': loss,
                f'{prefix}_property_loss': property_loss,
                f'{prefix}_natom_accuracy': num_atom_accuracy,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_volumes_mard': volumes_mard,
                f'{prefix}_type_accuracy': type_accuracy,
            })  # 附加评估指标

        return log_dict, loss


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    """根据配置创建模型并返回"""
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )
    return model


if __name__ == "__main__":
    main()  # 仅在脚本执行时运行
