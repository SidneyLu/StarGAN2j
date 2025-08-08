from os.path import join as ospj
import json
import os
import jittor as jt
import jittor.nn as nn

"""This is the module for auxiliary functions"""
"""辅助功能模块"""


def save_json(json_file, filename):
    """将JSON数据保存到文件"""
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    """打印网络结构和参数数量"""
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    """
    对特定类型的网络层应用He初始化

    参数:
        module: 要初始化的网络层
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x):
    """
    将归一化的图像数据还原到[0,1]范围

    参数:
        x: 输入的图像张量，通常范围为[-1,1]

    返回:
        范围在[0,1]之间的图像张量
    """
    out = (x + 1) / 2
    out.clamp_(0, 1)
    return out


def save_image(x, ncol, filename):
    """
    保存图像张量到文件

    参数:
        x: 图像张量
        ncol: 网格中的列数
        filename: 保存的文件名
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    x = denormalize(x)
    jt.save_image(x, filename, nrow=ncol, padding=0)


def lerp(input, end, weight):
    """
    线性插值，功能等同于torch.lerp

    参数:
        input: 起始值
        end: 结束值
        weight: 插值权重

    返回:
        插值结果
    """
    return input + weight * (end - input)


def translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename):
    """
    基于参考图像进行风格转换并重建原始图像

    参数:
        nets: 网络模型集合
        args: 参数配置
        x_src: 源图像
        y_src: 源图像标签
        x_ref: 参考图像
        y_ref: 参考图像标签
        filename: 保存结果的文件名
    """
    N, C, H, W = x_src.shape

    s_ref = nets.style_encoder(x_ref, y_ref)
    masks = None
    x_fake = nets.generator(x_src, s_ref, masks=masks)

    s_src = nets.style_encoder(x_src, y_src)
    masks = None
    x_rec = nets.generator(x_fake, s_src, masks=masks)

    x_concat = [x_src, x_ref, x_fake, x_rec]
    x_concat = jt.concat(x_concat, dim=0)

    save_image(x_concat, N, filename)
    del x_concat  # 释放内存


def translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename):
    """
    使用潜在空间向量进行风格转换

    参数:
        nets: 网络模型集合
        args: 参数配置
        x_src: 源图像
        y_trg_list: 目标风格标签列表
        z_trg_list: 潜在空间向量列表
        psi: 风格混合系数
        filename: 保存结果的文件名
    """
    N, C, H, W = x_src.shape
    latent_dim = z_trg_list[0].shape[1]
    x_concat = [x_src]
    masks = None

    for i, y_trg in enumerate(y_trg_list):
        z_many = jt.randn(10000, latent_dim)
        y_many = jt.full((10000,), y_trg[0], dtype='int32')
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = jt.mean(s_many, dim=0, keepdims=True)
        s_avg = s_avg.repeat(N, 1)

        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = lerp(s_avg, s_trg, psi)
            x_fake = nets.generator(x_src, s_trg, masks=masks)
            x_concat += [x_fake]

    x_concat = jt.concat(x_concat, dim=0)
    save_image(x_concat, N, filename)


def translate_using_reference(nets, args, x_src, x_ref, y_ref, filename):
    """
    使用参考图像进行风格转换

    参数:
        nets: 网络模型集合
        args: 参数配置
        x_src: 源图像
        x_ref: 参考图像
        y_ref: 参考图像标签
        filename: 保存结果的文件名
    """
    N, C, H, W = x_src.shape
    wb = jt.ones((1, C, H, W))
    x_src_with_wb = jt.concat([wb, x_src], dim=0)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    s_ref = nets.style_encoder(x_ref, y_ref)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = [x_src_with_wb]

    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        x_fake_with_ref = jt.concat([x_ref[i:i + 1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]

    x_concat = jt.concat(x_concat, dim=0)
    save_image(x_concat, N + 1, filename)
    del x_concat  # 释放内存


def debug_image(nets, args, inputs, step):
    """
    生成调试图像，包括循环一致性、潜在空间引导和参考引导的风格转换

    参数:
        nets: 网络模型集合
        args: 参数配置
        inputs: 输入数据
        step: 当前训练步骤
    """
    x_src, y_src = inputs.x_src, inputs.y_src
    x_ref, y_ref = inputs.x_ref, inputs.y_ref

    N = inputs.x_src.shape[0]

    # 循环一致性检查
    filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename)

    # 潜在空间引导的图像合成
    y_trg_list = [jt.full((N,), y, dtype='int32')
                  for y in range(min(args.num_domains, 5))]
    z_trg_list = jt.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1)

    for psi in [0.5, 0.7, 1.0]:
        filename = ospj(args.sample_dir, '%06d_latent_psi_%.1f.jpg' % (step, psi))
        translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename)

    # 参考引导的图像合成
    filename = ospj(args.sample_dir, '%06d_reference.jpg' % (step))
    translate_using_reference(nets, args, x_src, x_ref, y_ref, filename)