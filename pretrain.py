#!/usr/bin/env python3
import sys
#请根据自己路径修改
basedir="/home/xp/ATPnet"
sys.path.insert(1, basedir)
import random
import argparse
import json
import logging
import os
from time import time
from tqdm import tqdm
import datetime
import numpy as np
import pandas as pd
import torch
from APTAnet import APTAnet
from mydataset import ProteinSmileDataset
from transformers import BertModel, BertTokenizer
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve, roc_curve
)

#初始化随机数
def init_seeds(seed=0,cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def main(train_affinity_filepath, test_affinity_filepath, receptor_filepath,
        ligand_filepath,output_dir,training_name,pretrain_weights_path):
    # 初始化进程组
    torch.distributed.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device("cuda", local_rank)
    torch.cuda.empty_cache()
    # 初始化随机数
    init_seeds(123456+local_rank)
    # 设置log
    logging.basicConfig(level=logging.INFO if local_rank in [-1, 0] else logging.WARN)
    # 创建模型保存路径
    training_dir = os.path.join(output_dir, training_name)
    if not os.path.exists(training_dir) and local_rank==0:
        os.makedirs(training_dir)
    # 读入并在training_dir生成参数文件
    params = {}
    with open(os.path.join(output_dir, 'experiment/model_params.json')) as fp:
        params.update(json.load(fp))
    with open(os.path.join(training_dir, 'experiment/model_params.json'), 'w') as fp:
        json.dump(params, fp, indent=4)

    # 创建模型
    logging.info("Restore model...")
    model = APTAnet(params)
    # Restore model
    if pretrain_weights_path: 
    	model.load(pretrain_weights_path, map_location=device)

    # SyncBN
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    logging.info("DDP training")
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)

    num_params = sum(p.numel() for p in model.parameters())
    num_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of parameters: {num_params}, trainable: {num_train}')

    #读入两种预训练模型
    Protein_model_name="Rostlab/prot_bert_bfd"
    SMILES_model_name = "DeepChem/ChemBERTa-77M-MLM"
    # AA model
    Protein_tokenizer = BertTokenizer.from_pretrained(Protein_model_name, do_lower_case=False)
    Protein_model = BertModel.from_pretrained(Protein_model_name, torch_dtype=torch.float16)
    Protein_model = Protein_model.to(device)

    # SMILES model
    SMILES_tokenizer = RobertaTokenizer.from_pretrained(SMILES_model_name, local_files_only=True)
    SMILES_model = RobertaModel.from_pretrained(SMILES_model_name, torch_dtype=torch.float16, local_files_only=True)
    SMILES_model = SMILES_model.to(device)

    Protein_num_params = sum(p.numel() for p in Protein_model.parameters())
    SMILES_num_params = sum(p.numel() for p in SMILES_model.parameters())
    logging.info(f'Number of parameters Protein: {Protein_num_params}, SMILES: {SMILES_num_params}')

    #定义dataset数据集
    train_dataset = ProteinSmileDataset(
                 affinity_filepath=train_affinity_filepath,
                 receptor_filepath= receptor_filepath,
                 Protein_model=Protein_model,
                 Protein_tokenizer=Protein_tokenizer,
                 Protein_padding=params.get("receptor_padding_length"),
                 ligand_filepath=ligand_filepath,
                 SMILES_model=SMILES_model,
                 SMILES_tokenizer=SMILES_tokenizer,
                 SMILES_padding=params.get("ligand_padding_length"),
                 SMILES_argument=True,
                 SMILES_Canonicalization=False,
                 device=device
                 )
    train_sample = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=params['batch_size'],
        # shuffle=True,
        drop_last=False,
        num_workers=params.get('num_workers', 0),
        sampler=train_sample,
        pin_memory=False
    )
    test_dataset = ProteinSmileDataset(
        affinity_filepath=test_affinity_filepath,
        receptor_filepath=receptor_filepath,
        Protein_model=Protein_model,
        Protein_tokenizer=Protein_tokenizer,
        Protein_padding=params.get("receptor_padding_length"),
        ligand_filepath=ligand_filepath,
        SMILES_model=SMILES_model,
        SMILES_tokenizer=SMILES_tokenizer,
        SMILES_padding=params.get("ligand_padding_length"),
        SMILES_argument=False,
        SMILES_Canonicalization=True,
        device=device
    )
    test_sample = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=params['batch_size'],
        # shuffle=False,
        drop_last=False,
        num_workers=params.get('num_workers', 0),
        sampler=test_sample,
        pin_memory=False
    )

    logging.info(
        f'Training dataset has {len(train_dataset)} samples, '
        f'Testset  dataset has {len(test_dataset)} samples.'
    )
    logging.info(f'batchsize:{params["batch_size"]}')
    logging.info(f'num_workers:{params["num_workers"]}')
    logging.info(
        f'Loader length: Train - {len(train_loader)}, test - {len(test_loader)}'
    )

    # Define 优化器
    min_loss, max_roc_auc = 100, 0
    num_epochs = params.get('epochs')
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=params.get('lr'),
                                  weight_decay=params.get('weight_decay'),
                                  amsgrad=params.get('amsgrad'))
    
    # 学习率调整策略
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=params.get('lr'), pct_start=params.get("pct_start"),
                                                    steps_per_epoch=len(train_loader), anneal_strategy= params.get("anneal_strategy"),
                                                    epochs=num_epochs, verbose=False,three_phase=True)
    

    #模型保存方法
    save_top_model = os.path.join(training_dir, '{}_{}_{}.pt')
    def save(local_rank, path, metric, typ, val=None):
        """Routine to save model"""
        if local_rank == 1:
            save_name=path.format(typ, metric, round(val,4))
            model.module.save(save_name)
            if typ == 'best':
                logging.info(
                    f'\t New best performance in "{metric}"'
                    f' with value : {val} in epoch: {epoch}'
                )

    #开始训练
    logging.info('Training about to start...\n')
    result = []
    learning_rates=[params.get("lr")]
    for epoch in range(num_epochs):
        logging.info(f"== Epoch [{epoch}/{num_epochs}] ==")
        t = time()
        # Now training
        model.train()
        train_loader.sampler.set_epoch(epoch)
        train_loss = 0
        for ind, (receptor_seq,ligand_AA_seq,ligand_SMILES_seq,receptor_embedding,SMILES_embedding,y) in enumerate(train_loader):
            y_hat, pred_dict = model(SMILES_embedding.to(device), receptor_embedding.to(device))
            loss = model.module.loss(y_hat, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # change LR
            scheduler.step()
            learning_rates.append(optimizer.param_groups[0]["lr"])
            train_loss += loss.item()

        logging.info(
            "\t **** TRAINING ****   "
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"loss: {train_loss / len(train_loader):.5f}. "
            f"This took {time() - t:.1f} secs."
        )

        #test
        model.eval()
        test_loader.sampler.set_epoch(epoch)
        with torch.no_grad():
            test_loss = 0
            predictions = []
            labels = []
            for ind, (receptor_seq,ligand_AA_seq,ligand_SMILES_seq,receptor_embedding,SMILES_embedding,y) in enumerate(test_loader):
                y_hat, pred_dict = model(
                    SMILES_embedding.to(device), receptor_embedding.to(device)
                )
                predictions.append(y_hat)
                labels.append(y.clone())
                loss = model.module.loss(y_hat, y.to(device))
                test_loss += loss.item()
        predictions = torch.cat(predictions, dim=0).flatten().cpu().numpy()
        labels = torch.cat(labels, dim=0).flatten().cpu().numpy()

        test_loss = test_loss / len(test_loader)
        fpr, tpr, _ = roc_curve(labels, predictions)
        test_roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(labels, predictions)
        avg_precision = average_precision_score(labels, predictions)

        logging.info(
            f"\t **** TESTING **** Epoch [{epoch + 1}/{num_epochs}], "
            f"loss: {test_loss:.5f}, ROC-AUC: {test_roc_auc:.3f}, "
            f"Average precision: {avg_precision:.3f}."
        )

        if test_roc_auc > max_roc_auc:
            max_roc_auc = test_roc_auc
            save(local_rank, save_top_model, 'ROC-AUC', 'best', max_roc_auc)

        train_loss = train_loss / len(train_loader)
        result.append([epoch, test_loss, test_roc_auc, avg_precision, train_loss])

    #记录训练结果
    result = pd.DataFrame(result, columns=["epoch", "test_loss", "test_roc_auc", "avg_precision", "train_loss"])
    result.to_csv(os.path.join(training_dir, 'overview.csv'))



if __name__ == '__main__':
    #以soft split的fold1训练为例
    fold=1
    today=datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    #soft split
    train_affinity_filepath = basedir+'/data/tcr_split/fold'+str(fold)+'/train+covid.csv'
    test_affinity_filepath = basedir+'/data/tcr_split/fold'+str(fold)+'/test+covid.csv'
    receptor_filepath = basedir+'/data/tcr_full.csv'
    ligand_filepath = basedir+'/data/epitopes_merge.csv'
    output_dir= basedir+'/mytrain/experiment'
    training_name="CV"+str(fold)+"_soft_"+str(today)
    print(training_name)
    #是否加载预训练权重
    pretrain_weights_path=None
    main(train_affinity_filepath, test_affinity_filepath, receptor_filepath,
         ligand_filepath, output_dir, training_name,pretrain_weights_path)

    #CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --node_rank=0  pretrain.py
