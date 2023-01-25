from transformers import AutoModel, ViTModel, ViTFeatureExtractor
import torch.nn as nn
import torch

class TourClassifier(nn.Module):
  def __init__(self, n_classes1, n_classes2, n_classes3, text_model_name, image_model_name, device):
    super(TourClassifier, self).__init__()
    self.device = device
    self.text_model = AutoModel.from_pretrained(text_model_name).to(device)
    self.image_model = ViTModel.from_pretrained(image_model_name).to(device)
    self.text_model_name = text_model_name
    
    self.text_model.gradient_checkpointing_enable()  
    self.image_model.gradient_checkpointing_enable()  

    self.drop = nn.Dropout(p=0.1)

    def get_cls(input_size, target_size):
      return nn.Sequential(
          nn.Linear(input_size, self.text_model.config.hidden_size),
          nn.LayerNorm(self.text_model.config.hidden_size),
          nn.Dropout(p = 0.1),
          nn.ReLU(),
          nn.Linear(self.text_model.config.hidden_size, target_size),
      )
    
    def get_project(input_size, target_size):
      return nn.Sequential(
          nn.Linear(input_size, target_size),
          nn.ReLU(),
      )

    if text_model_name in ['monologg/kobert', 'bert-base-multilingual-uncased']:
        self.proj = get_project(self.image_model.config.hidden_size, self.text_model.config.hidden_size) # 1024 to 768
    self.cls = get_cls(self.text_model.config.hidden_size, n_classes1)
    self.cls2 = get_cls(self.text_model.config.hidden_size, n_classes2)
    self.cls3 = get_cls(self.text_model.config.hidden_size, n_classes3)
    
  def forward(self, input_ids, attention_mask, pixel_values):
    text_output = self.text_model(input_ids = input_ids, attention_mask=attention_mask)
    image_output = self.image_model(pixel_values = pixel_values)

    if self.text_model_name in ['monologg/kobert', 'bert-base-multilingual-uncased']:
        i_output = self.proj(image_output.last_hidden_state) # 1024 to 768
    else:
        i_output = image_output.last_hidden_state
    t_output = text_output.last_hidden_state
#     print(i_output.shape)
    concat_outputs = torch.cat([t_output, i_output], 1)
#     print(i_output.shape, concat_outputs.shape)
    # config hidden size 일치해야함
    encoder_layer = nn.TransformerEncoderLayer(d_model=self.text_model.config.hidden_size, nhead=8, batch_first=True).to(self.device)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to(self.device)
    outputs = transformer_encoder(concat_outputs)
    
    #cls token 
    outputs = outputs[:,0] # orig
#     outputs = torch.cat([outputs[:,0], outputs[:,256]], 1) # proj
    output = self.drop(outputs)

    out1 = self.cls(output)
    out2 = self.cls2(output)
    out3 = self.cls3(output)
    return out1,out2,out3
    