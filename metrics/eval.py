import os
import shutil
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import jittor as jt

from metrics.fid import calculate_fid_given_paths
from metrics.lpips import calculate_lpips_given_images
from core.data_loader import get_eval_loader
from core import utils


def calculate_metrics(nets, args, step, mode):
    print('\n===== 开始计算评估指标（模式：%s）=====' % mode)
    assert mode in ['latent', 'reference'], f"模式错误：必须为'latent'或'reference'"

    jt.flags.use_cuda = 1
    print(f"使用设备：{'GPU'}")

    domains = [d for d in os.listdir(args.val_img_dir) if os.path.isdir(os.path.join(args.val_img_dir, d))]
    domains.sort()
    num_domains = len(domains)
    print(f"检测到有效域数量：{num_domains}（{domains}）")
    if num_domains < 2:
        print("警告：域数量不足2个，直接返回")
        return

    lpips_dict = OrderedDict()

    for trg_idx, trg_domain in enumerate(domains):
        src_domains = [x for x in domains if x != trg_domain]
        print(f"\n----- 目标域：{trg_domain}（{trg_idx + 1}/{num_domains}）-----")

        if mode == 'reference':
            path_ref = os.path.join(args.val_img_dir, trg_domain)
            loader_ref = get_eval_loader(
                root=path_ref,
                img_size=args.img_size,
                batch_size=args.val_batch_size,
                imagenet_normalize=False,
                drop_last=True
            )
            print(f"加载参考图像：{path_ref}（批次大小：{args.val_batch_size}）")

        for src_idx, src_domain in enumerate(src_domains):
            task = f"{src_domain}2{trg_domain}"
            path_src = os.path.join(args.val_img_dir, src_domain)
            print(f"\n===== 处理任务：{task}（源域：{src_domain}）=====")

            loader_src = get_eval_loader(
                root=path_src,
                img_size=args.img_size,
                batch_size=args.val_batch_size,
                imagenet_normalize=False,
                drop_last=True
            )
            print(f"源域数据路径：{path_src}，总批次：{len(loader_src)}")

            path_fake = os.path.join(args.eval_dir, task)
            shutil.rmtree(path_fake, ignore_errors=True)
            os.makedirs(path_fake)

            lpips_values = []
            if mode == 'reference':
                iter_ref = iter(loader_ref)

            for batch_idx, x_src in enumerate(tqdm(loader_src, desc=f"生成 {task} 图像")):
                if not isinstance(x_src, jt.Var):
                    x_src = jt.array(x_src)
                N = x_src.shape[0]
                y_trg = jt.array([trg_idx] * N, dtype='int32')

                # 2. 生成掩码（如需）
                masks = None
                if args.w_hpf > 0:
                    masks = nets.fan.get_heatmap(x_src)
                    masks = masks.detach()


                group_of_images = []
                for out_idx in range(args.num_outs_per_domain):
                    s_trg = None
                    if mode == 'latent':
                        z_trg = jt.randn(N, args.latent_dim)
                        s_trg = nets.mapping_network(z_trg, y_trg)
                        del z_trg
                    else:
                        try:
                            x_ref = next(iter_ref)
                        except:
                            iter_ref = iter(loader_ref)
                            x_ref = next(iter_ref)

                        if not isinstance(x_ref, jt.Var):
                            x_ref = jt.array(x_ref)

                        if x_ref.shape[0] > N:
                            x_ref = x_ref[:N]
                        s_trg = nets.style_encoder(x_ref, y_trg)
                        del x_ref

                    x_fake = nets.generator(x_src, s_trg, masks=masks)
                    x_fake = x_fake.detach()
                    group_of_images.append(x_fake)

                    for img_idx in range(N):
                        img_tensor = x_fake[img_idx]  # [3, H, W]
                        img_path = os.path.join(
                            path_fake,
                            f"batch_{batch_idx:06d}_img_{img_idx:06d}_out_{out_idx:06d}.png"
                        )
                        utils.save_image(img_tensor, ncol=1, filename=img_path)
                    del x_fake

                lpips_value = calculate_lpips_given_images(group_of_images)
                lpips_values.append(lpips_value)
                del group_of_images

                del s_trg, masks

            del loader_src
            if mode == 'reference':
                del iter_ref

            if lpips_values:
                lpips_mean = np.mean(lpips_values)
                lpips_dict[f"LPIPS_{mode}/{task}"] = lpips_mean
                print(f"任务 {task} 平均LPIPS：{lpips_mean:.4f}")
            del lpips_values

        if mode == 'reference':
            del loader_ref
        jt.gc()

    if lpips_dict:
        lpips_mean_all = np.mean(list(lpips_dict.values()))
        lpips_dict[f"LPIPS_{mode}/mean"] = lpips_mean_all
        print(f"\n所有任务平均LPIPS：{lpips_mean_all:.4f}")
        lpips_path = os.path.join(args.eval_dir, f"LPIPS_{step:05d}_{mode}.json")
        utils.save_json(lpips_dict, lpips_path)
        print(f"LPIPS结果已保存至：{lpips_path}")
    del lpips_dict

    print("\n===== 开始计算FID指标 =====")
    calculate_fid_for_all_tasks(args, domains, step=step, mode=mode)
    print("\n===== 所有评估指标计算完成 =====")



def calculate_fid_for_all_tasks(args, domains, step, mode):
    fid_values = OrderedDict()
    for trg_domain in domains:
        src_domains = [x for x in domains if x != trg_domain]
        for src_domain in src_domains:
            task = f"{src_domain}2{trg_domain}"
            path_real = os.path.join(args.train_img_dir, trg_domain)
            path_fake = os.path.join(args.eval_dir, task)

            if not os.path.exists(path_fake) or len(os.listdir(path_fake)) == 0:
                print(f"警告：跳过任务 {task}，假图像目录为空")
                continue

            fid_batch_size = min(args.val_batch_size, 8)
            print(f"计算FID：{task}（批次大小：{fid_batch_size}）")
            fid_value = calculate_fid_given_paths(
                paths=[path_real, path_fake],
                img_size=args.img_size,
                batch_size=fid_batch_size
            )
            fid_values[f"FID_{mode}/{task}"] = fid_value


    if fid_values:
        fid_mean_all = np.mean(list(fid_values.values()))
        fid_values[f"FID_{mode}/mean"] = fid_mean_all
        print(f"所有任务平均FID：{fid_mean_all:.4f}")
        fid_path = os.path.join(args.eval_dir, f"FID_{step:05d}_{mode}.json")
        utils.save_json(fid_values, fid_path)
        print(f"FID结果已保存至：{fid_path}")
    del fid_values, domains