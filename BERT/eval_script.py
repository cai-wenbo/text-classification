from random import shuffle
from huggingface_hub import add_space_secret
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import transformers
from transformers import BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from utils.dataset import HotelDataset
import json
import os




def eval(eval_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


    '''
    load model
    '''
    #  If you have checked the source code of BertForSequenceClassification
    #  you will notice it works in the same way as a BERT+FNN, except it adds
    #  a dropout layer.
    if os.path.exists(eval_config['model_path_src']):
        model = BertForSequenceClassification.from_pretrained(eval_config['model_path_src'])
    else:
        print('Fine-tuned model not specified!')

    model = model.to(device)


    '''
    dataloader
    '''
    val_data  = HotelDataset('data/val.txt'  , max_length = 128)

    dataloader_val  = DataLoader(val_data  , batch_size = eval_config['batch_size'] , shuffle = False)



    '''
    train_loops
    '''
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)



    loss_sum_val = 0
    correct       = 0

    model.eval() 
    #  train loop
    for i, batch in enumerate(dataloader_val):
        batch = tuple(t.to(device) for t in batch)
        b_sentence_tensor, b_mask_tensor, b_label_tensor = batch
        
        with torch.no_grad():
            outputs = model(
                    b_sentence_tensor, 
                    token_type_ids = None,
                    attention_mask = b_mask_tensor
                    )

        predictions = torch.argmax(outputs[0], dim=1)
        correct += (predictions == b_label_tensor).sum().item()
        
    val_acc = correct / len(dataloader_val.dataset)


    print(f'Eval Accuracy: {val_acc:.6f}')





if __name__ == "__main__":
    eval_config                     = dict()
    eval_config['batch_size']       = 2
    eval_config['model_path_src']    = './saved_models/'
    eval(eval_config)
    

