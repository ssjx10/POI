from sklearn.model_selection import StratifiedKFold
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from transformers import AutoModel, ViTModel, AutoTokenizer, ViTFeatureExtractor, XLMRobertaTokenizer

from train_util  import *
from CateDataset import *
from model2 import TourClassifier
from tokenization_kobert import KoBertTokenizer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

os.makedirs("path", exist_ok=True)
os.makedirs("log", exist_ok=True)
device = torch.device("cuda")
df = pd.read_csv('data/train_folds_200.csv')
u, count = np.unique(df['cat3'].values, return_counts=True)
count_sort_ind = np.argsort(-count)
append_list = count_sort_ind[14:]

# fold
seed_everything(20)
is_amp = True
model_name = 'roberta_kr_200v1r2'
if is_amp:
    model_name += '_amp'
for i in range(5):
#     if i > 3:
#         continue
    print('*' * 10)
    print(f'Fold {i}')
    print('*' * 10)
    train = df[df["kfold"] != i].reset_index(drop=True)
    valid = df[df["kfold"] == i].reset_index(drop=True)
    # augmentation
    arr = []
    for k in range(len(train['back_en'])):
        if train['cat3'][k] in append_list and train['back_en'][k] == train['back_en'][k]: # version1
            b = train.loc[k].to_numpy()
            b[2], b[-2] = b[-2], b[2]
            arr.append(b)
#         if count[train['cat3'][k]] < 100: # version2
#             if train['back_en'][k] == train['back_en'][k]:
#                 b = train.loc[k].to_numpy()
#                 b[2], b[-2] = b[-2], b[2] # overview en
#                 arr.append(b[:])
#                 b[2], b[-1] = b[-1], b[2] # overview ja
#                 arr.append(b[:])
#         elif count[train['cat3'][k]] < 300:
#             if train['back_en'][k] == train['back_en'][k]:
#                 b = train.loc[k].to_numpy()
#                 b[2], b[-2] = b[-2], b[2]
#                 arr.append(b)
    df2 = pd.DataFrame(arr, columns=train.columns)
    train = pd.concat([train, df2], ignore_index = True)
    print(f'train len : {len(train)}')
    transformer_model = 'klue/roberta-large'
#     transformer_model = 'xlm-roberta-large'
    log_name = f'log/{model_name}_fold{i}.txt'
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % transformer_model)  # save the log
    vit_model = 'google/vit-large-patch32-384'
    if transformer_model == 'monologg/kobert':
        tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    else:
        tokenizer = AutoTokenizer.from_pretrained(transformer_model)
    vit_model = 'google/vit-large-patch32-384'
    tokenizer = AutoTokenizer.from_pretrained(transformer_model)
    feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model)
    train_data_loader = create_data_loader(train, tokenizer, feature_extractor, 256, 32, shuffle_=True)
    valid_data_loader = create_data_loader(valid, tokenizer, feature_extractor, 256, 32, is_train=False)
    
    epochs = 30
    if is_amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
    model = TourClassifier(n_classes1 = 6, n_classes2 = 18, n_classes3 = 128,
                           text_model_name = transformer_model, image_model_name = vit_model, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr= 3e-5)
    total_steps = len(train_data_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps = int(total_steps * 0.1),
    num_training_steps = total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    max_acc = 0
    for epoch in range(epochs):
        print('-' * 10)
        print(f'Epoch {epoch}/{epochs - 1}')
        print('-' * 10)
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train),
            epoch, is_amp=is_amp, scaler=scaler
        )
        validate_acc, validate_loss = validate(
            model,
            valid_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(valid), is_amp=is_amp
        )

        if validate_acc[2] > max_acc:
            max_acc = validate_acc[2]
            torch.save(model.state_dict(), f'path/{model_name}_fold{i}.pt')

        mesg = f'Train loss {train_loss} accuracy {train_acc}\n'
        mesg += f'Validate loss {validate_loss} accuracy {validate_acc}'
        print(mesg)
        mesg = f'Epoch {epoch}/{epochs - 1}' + mesg
        log_name = f'log/{model_name}_fold{i}.txt'
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % str(mesg))  # save the log
        print("")