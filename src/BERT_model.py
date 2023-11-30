from BERT_classifier import *

from transformers import BertTokenizer, get_linear_schedule_with_warmup

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW

import numpy as np


def BERT_model(data):
    #Split the training and test data into 80/20 split
    train_pct = .8
    np.random.seed(1)
    idx = np.random.permutation(len(data))

    X_train = data['Preprocessed_Text_No_Stopwords'].values[idx[:int(train_pct * len(data))]]
    y_train = data['Sentiment'].values[idx[:int(train_pct * len(data))]]
    y_train[y_train == -1] = 0

    X_test = data['Preprocessed_Text_No_Stopwords'].values[idx[int(train_pct * len(data)):]]
    y_test = data['Sentiment'].values[idx[int(train_pct * len(data)):]]
    y_test[y_test == -1] = 0

    #Prepare the Bert NLP model tokenizer to encode tweets
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Use this to determine max length for encoding, (48)
    encoded = [tokenizer.encode(sent, add_special_tokens=True) for sent in data['Preprocessed_Text_No_Stopwords'].values]
    MAX_LEN = max([len(sent) for sent in encoded])
    print('Max length: ', MAX_LEN)

    #Encode training dataset for BERT

    X_train_inputs = []
    X_train_masks = []

    #For each tweet (train)
    for line in X_train:
        # encode the data. Return input encoding and attention mask
        encoding = tokenizer.encode_plus(
            text=line,  # data to process
            add_special_tokens=True,  # adds special chars [CLS] and [SEP] to encoding
            padding='max_length',  # pad the tweets with 0s to fit max length
            max_length=48,  # assign max length
            truncation=True,  # truncate tweets longer than max length
            return_tensors="pt",  # return tensor as pytorch tensor
            return_attention_mask=True  # return the attention mask
        )

        # add the encodings to the list
        X_train_inputs.append(encoding.get('input_ids'))
        X_train_masks.append(encoding.get('attention_mask'))

    #Return the lists as tensors
    X_train_inputs = torch.concat(X_train_inputs)
    X_train_masks = torch.concat(X_train_masks)


    #Encode testing dataset for BERT
    X_test_inputs = []
    X_test_masks = []

    #For each tweet (test)
    for line in X_test:
        # encode the data. Return input encoding and attention mask
        encoding = tokenizer.encode_plus(
            text=line,  # data to process
            add_special_tokens=True,  # adds special chars [CLS] and [SEP] to encoding
            padding='max_length',  # pad the tweets with 0s to fit max length
            max_length=48,  # assign max length
            truncation=True,  # truncate tweets longer than max length
            return_tensors="pt",  # return tensor as pytorch tensor
            return_attention_mask=True  # return the attention mask
        )

        # add the encodings to the list
        X_test_inputs.append(encoding.get('input_ids'))
        X_test_masks.append(encoding.get('attention_mask'))

    #Return the lists as tensors
    X_test_inputs = torch.concat(X_test_inputs)
    X_test_masks = torch.concat(X_test_masks)


    #Get the train and test labels
    y_train_labels = torch.tensor(y_train)
    y_test_labels = torch.tensor(y_test)

    print(X_train_inputs.shape, X_train_masks.shape, y_train_labels.shape)
    print(X_test_inputs.shape, X_test_masks.shape, y_test_labels.shape)

    #Set batch size to 16. recommended 16 or 32 depending on GPU size
    batch_size = 16

    #Randomize the train data and define dataloader for model training
    train_data = TensorDataset(X_train_inputs, X_train_masks, y_train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    #Randomize the test data and define dataloader for model testing
    test_data = TensorDataset(X_test_inputs, X_test_masks, y_test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    #Set random seed for repeatability
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    # Check if GPU is available and assign device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Initialize Bert Classifier
    model = BertClassifier(freeze=False)

    # Send model to device (GPU if available)
    model.to(device)

    # Define model hyperparameters
    epochs = 4
    steps = len(train_dataloader) * epochs
    learning_rate = 5e-5
    epsilon = 1e-8

    # Define Adam optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)

    # Define scheduler for training the optimizer
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=steps)

    # Define cross entropy loss function
    loss_function = nn.CrossEntropyLoss()

    # For the number of epochs
    for e in range(epochs):
        # Assign model to train
        model.train()

        # Intialize loss to zero
        train_loss = 0

        # For each batch
        for batch in train_dataloader:
            # Get batch inputs, masks and labels
            batch_inputs, batch_masks, batch_labels = batch

            # Send variables to device (GPU if available)
            batch_inputs = batch_inputs.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)

            # Reset the model gradient
            model.zero_grad()

            #Get classification of encoded values
            logits = model(batch_inputs, batch_masks)

            #Calculate loss based on predictions and known values
            loss = loss_function(logits, batch_labels)

            #Add loss to the running total
            train_loss += loss.item()

            #Update the model weights based on the loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        #Calculate the average loss over batch
        train_loss /= len(train_dataloader)

        #Assign the model to evaluate
        model.eval()

        #Initialize losses
        test_loss = 0
        test_acc = 0

        #For each batch
        for batch in test_dataloader:
            #Get encoding inputs, masks and labels
            batch_inputs, batch_masks, batch_labels = batch

            #Send variables to device (GPU if available)
            batch_inputs = batch_inputs.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)

            #Predict the input values without updating the model
            with torch.no_grad():
                logits = model(batch_inputs, batch_masks)

            #Calculate the loss
            loss = loss_function(logits, batch_labels)
            test_loss += loss.item()

            #Convert predictions to 0 and 1
            preds = torch.argmax(logits, dim=1).flatten()

            #Calculate accuracy of model on test data
            accuracy = (preds == batch_labels).cpu().numpy().mean() * 100
            test_acc += accuracy

        #Calculate average loss and accuracy per each batch
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

        #Print epoch information
        print('Epoch: %d  |  Train Loss: %1.5f  |  Test Loss: %1.5f  |  Test Accuracy: %1.2f' % (
        e + 1, train_loss, test_loss, test_acc))

    #Save model
    torch.save(model.state_dict(), '../outputs/stock_sentiment_model.pt')


















