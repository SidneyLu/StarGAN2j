from os.path import join as ospj
import json

import jittor
import jittor.nn as nn

"""This is the module for auxiliary functions"""
"""辅助功能模块"""

def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)

def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    print("Number of parameters of %s: %i" % (name, num_params))

#Initializing weights for specific network layers
#为特定类型的网络层初始化权重
def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

#图像数据标准化还原
def denormalize(x):
    out = (x + 1) / 2
    out = jittor.array(out)
    out.clamp_(0, 1)
    return out

#保存图像
def save_image(x, ncol, filename):
    x = denormalize(x)
    print(x.shape)
    jittor.save_image(x, filename, nrow=ncol, padding=0)

#Just as torch.lerp
#对齐torch.lerp函数的功能
def lerp(input, end, weight, out=None):
    out = input + weight * (end - input)
    return out


@jittor.no_grad()
def translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.shape

    s_ref = nets.style_encoder(x_ref, y_ref)
    masks = None
    x_fake = nets.generator(x_src, s_ref, masks=masks)

    s_src = nets.style_encoder(x_src, y_src)
    masks = None
    x_rec = nets.generator(x_fake, s_src, masks=masks)

    x_concat = [x_src, x_ref, x_fake, x_rec]
    x_concat = jittor.concat(x_concat, dim=0)

    save_image(x_concat, N, filename)
    del x_concat

@jittor.no_grad()
def translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename):
    N, C, H, W = x_src.shape
    latent_dim = z_trg_list[0].size(1)
    x_concat = [x_src]
    masks = None

    for i, y_trg in enumerate(y_trg_list):
        z_many = jittor.randn(10000, latent_dim)
        y_many = jittor.var(10000).fill_(y_trg[0])
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = jittor.mean(s_many, dim=0, keepdims=True)
        s_avg = s_avg.repeat(N, 1)

        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = lerp(s_avg, s_trg, psi)
            x_fake = nets.generator(x_src, s_trg, masks=masks)
            x_concat += [x_fake]

    x_concat = jittor.concat(x_concat, dim=0)
    save_image(x_concat, N, filename)

@jittor.no_grad()
def translate_using_reference(nets, args, x_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.shape
    wb = jittor.ones((1, C, H, W))
    x_src_with_wb = jittor.concat([wb, x_src], dim=0)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    s_ref = nets.style_encoder(x_ref, y_ref)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = [x_src_with_wb]
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        x_fake_with_ref = jittor.concat([x_ref[i:i+1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]

    x_concat = jittor.concat(x_concat, dim=0)
    save_image(x_concat, N+1, filename)
    del x_concat

#Generating samples while training
#训练过程中生成示例图像
@jittor.no_grad()
def debug_image(nets, args, inputs, step):
    x_src, y_src = inputs.x_src, inputs.y_src
    x_ref, y_ref = inputs.x_ref, inputs.y_ref

    N = inputs.x_src.shape[0]

    #Translate and reconstruct (reference-guided)
    filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename)

    #Latent-guided image synthesis
    y_trg_list = [jittor.var(y).repeat(N)
                  for y in range(min(args.num_domains, 5))]
    z_trg_list = jittor.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1)
    for psi in [0.5, 0.7, 1.0]:
        filename = ospj(args.sample_dir, '%06d_latent_psi_%.1f.jpg' % (step, psi))
        translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename)

    #Reference-guided image synthesis
    filename = ospj(args.sample_dir, '%06d_reference.jpg' % (step))
    translate_using_reference(nets, args, x_src, x_ref, y_ref, filename)