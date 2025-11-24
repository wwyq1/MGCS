# Noisy Correspondence Learning with Modality Gap Direction Correction-AAAI2026
## 文章摘要：Cross-modal retrieval is crucial for discovering latent correspondences across different modalities. However, existing methods typically assume that training data are well-aligned, an unrealistic assumption since real-world datasets inevitably contain noisy correspondences. Many current approaches attempt to handle noise using strategies borrowed from single-modal classification, such as the small-loss trick, to identify clean training pairs. However, our experiments reveal that such small-loss-based strategies are less effective for multi-modal tasks due to the inherent modality gaps. Through comprehensive analysis, we observe that the deviation directions between paired image-caption features, termed Sample-level Alignment Drift (SAD), are compact and data-dependent. Leveraging this discovery, we introduce the Modality Gap Corrected Similarity (MGCS) framework that can more accurately measure the semantic distances of cross-modal samples, dynamically compensating for misalignment. Within MGCS, we can achieve more reliable noisy data separation to promote correct supervision during cross-modal matching model training. Extensive experiments on three widely used noisy correspondence benchmarks demonstrate that MGCS significantly surpasses current state-of-the-art methods. 
## 训练步骤
1. 运行环境配置，数据集下载以及命令运行建议参考 https://github.com/xu5zhao/BiCro
2. 下面给出一个作者在自己服务器上运行命令的示例供参考，此命令即为在f30k数据集以百分之20的噪声环境进行运行：
   ```
   python run.py --gpu 0 --workers 4  --data_name f30k_precomp --noise_ratio 0.2 --data_path "/root/autodl-tmp/YQ/download/data/" --vocab_path "/root/autodl-tmp/YQ/download/vocab/f30k/" --seed 96 --warmup_epoch 10 --warmup_type warmup_sele --id train_f30k --warmup_rate 0.5 --p_threshold 0.5 --noise_train train --noise_tem 0.5 --fit_type bmm  --num_epochs 400 --batch_size 96 --soft_margin exponential --fit_type bmm  --grad_clip 2.0  --learning_rate 0.0002
   ```
3. 对于训练好的跨模态匹配模型的ckpt的评估指令，这里同样给出在作者自己服务器上运行命令仅供参考：
     ```
    python evaluation.py  --mode_path "/root/WW-XD/MGCS/output/2025_05_20_21_43_48/checkpoint_dev_best.pth.tar" --data_name f30k_precomp --data_path "/root/autodl-tmp/YQ/download/data/" --vocab_path "/root/autodl-tmp/YQ/download/vocab/f30k/"
     ```
     

