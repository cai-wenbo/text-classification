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




def train(training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


    '''
    load model and the history
    '''
    #  If you have checked the source code of BertForSequenceClassification
    #  you will notice it works in the same way as a BERT+FNN, except it adds
    #  a dropout layer.
    model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels = 2)
    if training_config.get('model_path_src') is not None:
        model = BertForSequenceClassification.from_pretrained(training_config['saved_embedding'], num_labels = 2)

    model = model.to(device)

    #  load the losses history
    step_losses = list()
    if os.path.exists(training_config['step_losses_pth']):
        with open(training_config['step_losses'], 'r') as file:
            step_losses = json.load(file)
            file.close()

    train_losses = list()
    if os.path.exists(training_config['train_losses_pth']):
        with open(training_config['train_losses'], 'r') as file:
            train_losses = json.load(file)
            file.close()
    
    test_losses = list()
    if os.path.exists(training_config['test_losses_pth']):
        with open(training_config['test_losses'], 'r') as file:
            test_losses = json.load(file)
            file.close()

    '''
    dataloader
    '''
    train_data = HotelDataset('data/train.txt' , max_length = 128)
    test_data  = HotelDataset('data/test.txt'  , max_length = 128)

    dataloader_train = DataLoader(train_data , batch_size = training_config['batch_size'] , shuffle = True)
    dataloader_test  = DataLoader(test_data  , batch_size = training_config['batch_size'] , shuffle = False)


    '''
    optimizer
    '''
    ##@title Optimizer Grouped Parameters
    #This code is taken from:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L102

    # Don't apply weight decay to any parameters whose names include these tokens.
    # (Here, the BERT doesn't have `gamma` or `beta` parameters, only `bias` terms)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    # Separate the `weight` parameters from the `bias` parameters.
    # - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01.
    # - For the `bias` parameters, the 'weight_decay_rate' is 0.0.
    optimizer_grouped_parameters = [
        # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.1},

        # Filter for parameters which *do* include those.
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    # Note - `optimizer_grouped_parameters` only includes the parameter values, not
    # the names.

    optimizer = AdamW(
            optimizer_grouped_parameters, 
            lr = training_config['learning_rate'],
            eps = 1e-8
            )

    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = 0, # Default value in run_glue.py
            num_training_steps = len(dataloader_train) * training_config['num_of_epochs']
            )


    '''
    creterian
    '''
    creterian = nn.CrossEntropyLoss(reduction='mean').to(device)



    '''
    train_loops
    '''
    for epoch in range(training_config['num_of_epochs']):
        loss_sum_train = 0
        loss_sum_test  = 0
        
        model.train()
        #  train loop
        for i, batch in enumerate(dataloader_train):
            batch = tuple(t.to(device) for t in batch)
            b_sentence_tensor, b_mask_tensor, b_label_tensor = batch

            optimizer.zero_grad()

            outputs = model(
                    b_sentence_tensor, 
                    token_type_ids = None,
                    attention_mask = b_mask_tensor
                    )

            loss = creterian(outputs[0].view(-1, 2), b_label_tensor.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum_train += torch.sum(loss)
            step_losses.append(loss)



        model.eval() 
        #  train loop
        for i, batch in enumerate(dataloader_test):
            batch = tuple(t.to(device) for t in batch)
            b_sentence_tensor, b_mask_tensor, b_label_tensor = batch
            
            with torch.no_grad():
                outputs = model(
                        b_sentence_tensor, 
                        token_type_ids = None,
                        attention_mask = b_mask_tensor
                        )

            loss = creterian(outputs[0].view(-1, 2), b_label_tensor.view(-1))
            loss_sum_test += torch.sum(loss)
            
        train_loss = loss_sum_train / len(dataloader_train)
        test_loss = loss_sum_test / len(dataloader_test)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')


    '''    
    save model and data
    '''

    model = model.to('cpu')
    model.save_pretrained(training_config['model_path_dst'])

    #  save the loss of the steps
    with open(training_config['step_losses_pth'], 'w') as file:
        json.dump(step_losses, file)
        file.close()

    with open(training_config['train_losses_pth'], 'w') as file:
        json.dump(train_losses, file)
        file.close()
    
    with open(training_config['test_losses_pth'], 'w') as file:
        json.dump(test_losses, file)
        file.close()



if __name__ == "__main__":
    training_config = dict()
    training_config['num_of_epochs']        = 5
    training_config['batch_size']           = 2
    training_config['model_path_dst']       = './saved_models/'
    #  training_config['checkpoint'] = 0
    training_config['learning_rate']        = 1e-4
    training_config['step_losses_pth']  = './step_losses.json'
    training_config['train_losses_pth']  = './train_losses.json'
    training_config['test_losses_pth']  = './test_losses.json'
    #  training_config['model_path_src']    = 'saved_embedding.pth'
    train(training_config)
    

