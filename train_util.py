import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
import argparse
import os, random, time
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
import torch.nn.functional as F
import torch.cuda.amp as amp
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm

from eval import *


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, epoch, loss_w=None, is_amp=False, scaler=None):
  
    batch_time = AverageMeter()     
    data_time = AverageMeter()      
    losses = AverageMeter()         
    accuracies = AverageMeter()
    f1_accuracies = AverageMeter()
    sent_count = AverageMeter()   

    start = end = time.time()

    model = model.train()
    correct_predictions = 0
    for step, d in enumerate(data_loader):
        data_time.update(time.time() - end)
        batch_size = d["input_ids"].size(0) 

        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        pixel_values = d['pixel_values'].to(device)
        cats1 = d["cats1"].to(device)
        cats2 = d["cats2"].to(device)
        cats3 = d["cats3"].to(device)
        
        optimizer.zero_grad()
        if is_amp:
            with amp.autocast():
                outputs,outputs2,outputs3 = model(
                  input_ids=input_ids,
                  attention_mask=attention_mask,
                  pixel_values=pixel_values
                )
                _, preds = torch.max(outputs3, dim=1)

                if loss_w is None:
                    loss1 = loss_fn(outputs, cats1)
                    loss2 = loss_fn(outputs2, cats2)
                    loss3 = loss_fn(outputs3, cats3)
                    loss = loss1 * 0.05 + loss2 * 0.1 + loss3 * 0.85
                else: # model2
                    loss1 = loss_fn(outputs, cats3)
                    loss2 = loss_fn(outputs2, cats2) # image
                    loss3 = loss_fn(outputs3, cats3)
                    loss = loss1 * loss_w[0] + loss2 * loss_w[1] + loss3 * loss_w[2]
                    
            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs,outputs2,outputs3 = model(
              input_ids=input_ids,
              attention_mask=attention_mask,
              pixel_values=pixel_values
            )
            _, preds = torch.max(outputs3, dim=1)

            if loss_w is None:
                loss1 = loss_fn(outputs, cats1)
                loss2 = loss_fn(outputs2, cats2)
                loss3 = loss_fn(outputs3, cats3)
                loss = loss1 * 0.05 + loss2 * 0.1 + loss3 * 0.85
            else: # model2
                loss1 = loss_fn(outputs, cats3)
                loss2 = loss_fn(outputs2, cats2)
                loss3 = loss_fn(outputs3, cats3)
                loss = loss1 * loss_w[0] + loss2 * loss_w[1] + loss3 * loss_w[2]
            
            losses.update(loss.item(), batch_size)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        correct_predictions += torch.sum(preds == cats3)
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)
        if step % 200 == 0 or step == (len(data_loader)-1):
                acc,f1_acc = calc_tour_acc(outputs3, cats3)
                accuracies.update(acc, batch_size)
                f1_accuracies.update(f1_acc, batch_size)


                print('Epoch: [{0}][{1}/{2}] '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                      'Elapsed {remain:s} '
                      'Loss: {loss.val:.3f}({loss.avg:.3f}) '
                      'Acc: {acc.val:.3f}({acc.avg:.3f}) '   
                      'f1_Acc: {f1_acc.val:.3f}({f1_acc.avg:.3f}) '           
                      'sent/s {sent_s:.0f} '
                      .format(
                      epoch, step+1, len(data_loader),
                      data_time=data_time, loss=losses,
                      acc=accuracies,
                      f1_acc=f1_accuracies,
                      remain=timeSince(start, float(step+1)/len(data_loader)),
                      sent_s=sent_count.avg/batch_time.avg
                      ))

    return correct_predictions.double() / n_examples, losses.avg

def validate(model,data_loader,loss_fn,optimizer,device,scheduler,n_examples, loss_w=None, is_amp=False):
    model = model.eval()
    losses = []
    corr_preds3 = 0
    cnt = 0
    for d in tqdm(data_loader):
        with torch.no_grad():
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            pixel_values = d['pixel_values'].to(device)
            cats1 = d["cats1"].to(device)
            cats2 = d["cats2"].to(device)
            cats3 = d["cats3"].to(device)
            
            if is_amp:
                with amp.autocast():
                    outputs, outputs2, outputs3 = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values
                    )

                    if loss_w is None:
                        loss1 = loss_fn(outputs, cats1)
                        loss2 = loss_fn(outputs2, cats2)
                        loss3 = loss_fn(outputs3, cats3)
                        loss = loss1 * 0.05 + loss2 * 0.1 + loss3 * 0.85
                    else:
                        loss1 = loss_fn(outputs, cats3)
                        loss2 = loss_fn(outputs2, cats2)
                        loss3 = loss_fn(outputs3, cats3)
                        loss = loss1 * loss_w[0] + loss2 * loss_w[1] + loss3 * loss_w[2]

                        cats1 = cats3
#                         cats2 = cats3
            else:
                outputs, outputs2, outputs3 = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
                )

                if loss_w is None:
                    loss1 = loss_fn(outputs, cats1)
                    loss2 = loss_fn(outputs2, cats2)
                    loss3 = loss_fn(outputs3, cats3)
                    loss = loss1 * 0.05 + loss2 * 0.1 + loss3 * 0.85
                else:
                    loss1 = loss_fn(outputs, cats3)
                    loss2 = loss_fn(outputs2, cats2)
                    loss3 = loss_fn(outputs3, cats3)
                    loss = loss1 * loss_w[0] + loss2 * loss_w[1] + loss3 * loss_w[2]

                    cats1 = cats3
#                     cats2 = cats3 # image
            
            _, preds3 = torch.max(outputs3, dim=1)
            corr_preds3 += torch.sum(preds3 == cats3)

            losses.append(loss.item())
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if cnt == 0:
                cnt +=1
                outputs1_arr = outputs
                outputs2_arr = outputs2
                outputs3_arr = outputs3
                cats1_arr = cats1
                cats2_arr = cats2
                cats3_arr = cats3
            else:
                outputs1_arr = torch.cat([outputs1_arr, outputs], 0)
                outputs2_arr = torch.cat([outputs2_arr, outputs2], 0)
                outputs3_arr = torch.cat([outputs3_arr, outputs3], 0)
                cats1_arr = torch.cat([cats1_arr, cats1], 0)
                cats2_arr = torch.cat([cats2_arr, cats2], 0)
                cats3_arr = torch.cat([cats3_arr, cats3], 0)
    acc1, f1_acc1 = calc_tour_acc(outputs1_arr, cats1_arr)
    acc2, f1_acc2 = calc_tour_acc(outputs2_arr, cats2_arr)
    acc3, f1_acc3 = calc_tour_acc(outputs3_arr, cats3_arr)
    return [f1_acc1, f1_acc2, f1_acc3], np.mean(losses)

def inference(model, data_loader, device, n_examples, is_amp=False):
    model = model.eval()
    preds_arr = []
    preds_arr2 = []
    preds_arr3 = []
    logits_arr = []
    for d in tqdm(data_loader):
        with torch.no_grad():
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            pixel_values = d['pixel_values'].to(device)
            
            if is_amp:
                with amp.autocast():
                    outputs,outputs2,outputs3 = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values
                    )
            else:
                outputs,outputs2,outputs3 = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
                )


            _, preds = torch.max(outputs, dim=1)
            _, preds2 = torch.max(outputs2, dim=1)
            _, preds3 = torch.max(outputs3, dim=1)
            logits = F.softmax(outputs3, dim=1)
            logits_arr.append(logits.cpu().numpy())

            preds_arr.append(preds.cpu().numpy())
            preds_arr2.append(preds2.cpu().numpy())
            preds_arr3.append(preds3.cpu().numpy())

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    return preds_arr, preds_arr2, preds_arr3, logits_arr
