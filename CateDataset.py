import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
from bs4 import BeautifulSoup
import re

def preprocess_sentence(sentence):
    sent = BeautifulSoup(sentence, "lxml").text # <br />, <a href = ...> 등의 html 태그 제거
    if sent == '':
        sent = re.sub(r'\<[^>]*\>', '', sentence) # <> 괄호로 닫힌 문자열  제거
    sentence = re.sub(r'\([^)]*\)', '', sent) # () 괄호로 닫힌 문자열  제거
    sentence = re.sub('[^ ㄱ-ㅣ가-힣]+', ' ', sentence) # 한글 추출
    sentence = re.sub('\n+', ' ', sentence)
    sentence = re.sub('\s+', ' ', sentence)
    sentence = re.sub('※', '', sentence)
    sentence = re.sub('#', '', sentence)
    sentence = re.sub('\*', '', sentence)
    sentence = re.sub('\'', '', sentence)
    sentence = sentence.lstrip()
    sentence = sentence.rstrip()
    
    return sentence

class CategoryDataset(Dataset):
  def __init__(self, text, back, image_path, cats1, cats2, cats3, tokenizer, feature_extractor, max_len, is_train = True):
    self.text = text
    self.back1 = back[0] # len 2 
    self.back2 = back[1]
    self.image_path = image_path
    self.cats1 = cats1
    self.cats2 = cats2
    self.cats3 = cats3
    self.tokenizer = tokenizer
    self.feature_extractor = feature_extractor
    self.max_len = max_len
    self.is_train = is_train

  def __len__(self):
    return len(self.text)

  def __getitem__(self, item):
    text = str(self.text[item])
    
    if self.is_train and self.back1[item] == self.back1[item]: # random choice
#         t_arr  = [text, str(self.back1[item]), str(self.back2[item])]
        t_arr  = [text, str(self.back1[item])] # overview, en
#         print('random len', len(t_arr))
        idx = np.random.choice(len(t_arr))
        text = t_arr[idx]
    text = preprocess_sentence(text)
    image_path = os.path.join('data', str(self.image_path[item])[2:])
    image = cv2.imread(image_path)
    cat = self.cats1[item]
    cat2 = self.cats2[item]
    cat3 = self.cats3[item]
    
    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding = 'max_length',
      truncation = True,
      return_attention_mask=True,
      return_tensors='pt',
    )
#     print(image_path)
    image_feature = self.feature_extractor(images=image, return_tensors="pt")
    return {
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'pixel_values': image_feature['pixel_values'][0],
      'cats1': torch.tensor(cat, dtype=torch.long),
      'cats2': torch.tensor(cat2, dtype=torch.long),
      'cats3': torch.tensor(cat3, dtype=torch.long)
    }

def create_data_loader(df, tokenizer, feature_extractor, max_len, batch_size, shuffle_=False, is_train = True, is_infer=False):
    if is_infer:
        is_train = False
    dummy = np.zeros_like(df.img_path.to_numpy())
    if is_infer:
        cat1 = dummy
        cat2 = dummy
        cat3 = dummy
        back = [df.overview.to_numpy(), df.overview.to_numpy()]
    else:
        cat1 = df.cat1.to_numpy()
        cat2 = df.cat2.to_numpy()
        cat3 = df.cat3.to_numpy()
        back = [df.back_en.to_numpy(), df.back_ja.to_numpy()]
        
    ds = CategoryDataset(
        text=df.overview.to_numpy(),
        back=back, # back
        image_path = df.img_path.to_numpy(),
        cats1=cat1,
        cats2=cat2,
        cats3=cat3,
        tokenizer=tokenizer,
        feature_extractor = feature_extractor,
        max_len=max_len, is_train=is_train
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle = shuffle_
    )