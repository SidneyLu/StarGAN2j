# StarGAN2j - A Jittor Implementation of StarGANv2

## StarGAN2j - StarGANv2官方仓库的个人复现（计图）
>CCF-GrokCV 新芽计划 第二阶段考核任务 \
鲁昕宁 南开大学 2414015@mail.nankai.edu.cn \
由于工作量和计算资源的限制，目前仅复现了基于AFHQ2数据集的模型训练、评估、加载、保存功能及图片生成功能

## Generation
```bash
python main.py --mode sample --resume_iter 10000 
```
Pretrained network is stored in expr/checkpoints

## Training 
### (on AFHQ2 Dataset)
Download AFHQ2 dataset from https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0 \
Unzip the .zip file and make your directory structure like this:  \
```
-data/afhq2/train \
-data/afhq2/test \
```
Then run
```bash
python main.py --mode train 
```
Generated images and network checkpoints will be stored in `expr/samples` and `expr/checkpoints` directories respectively. \
Total iterations: 10000  
Trained on RTX 2070 Super, single GPU, for 6 Days
## Evaluation
| Dataset | FID (latent) | LPIPS (latent) | FID (reference) | LPIPS (reference) | Elapsed time  |
|:--------|:------------:|:--------------:|:---------------:|:-----------------:|:-------------:|
| `AFHQ2` |   60.8967    |     0.3850     |     61.2043     |      0.3486       |    

run
```bash
python main.py --mode eval --resume_iter 10000
```
Your evaluation results will be saved into `expr/eval`

## Citation
```
@inproceedings{choi2020starganv2,
  title={StarGAN v2: Diverse Image Synthesis for Multiple Domains},
  author={Yunjey Choi and Youngjung Uh and Jaejun Yoo and Jung-Woo Ha},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## Acknowledgements
