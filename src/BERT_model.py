from BERT_classifier import *

from transformers import BertTokenizer, get_linear_schedule_with_warmup

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW

import numpy as np


def BERT_model(data):
    # Split the training and test data into 80/20 split
    train_percentage = 0.8
    np.random.seed(1)
    # Shuffling of indices to randomize the order of data samples during tasks such as dataset splitting or batch
    # processing, promoting better model training and generalization
    indices = np.random.permutation(len(data))

    # Select the first 80% of indices for training
    train_indices = indices[:int(train_percentage * len(data))]
    # Select the remaining 20% of indices for testing
    test_indices = indices[int(train_percentage * len(data)):]

    # Extract preprocessed text and sentiment labels for the training set
    train_texts = data['Preprocessed_Text_No_Stopwords'].values[train_indices]
    # Labels are the desired prediction or classification for each corresponding input.
    train_labels = data['Sentiment'].values[train_indices]
    # To adhere to a more standard convention where classes are typically numbered starting from 0,
    # we replace any -1 sentiment labels with 0
    train_labels[train_labels == -1] = 0

    # Extract preprocessed text and sentiment labels for the test set
    test_texts = data['Preprocessed_Text_No_Stopwords'].values[test_indices]
    test_labels = data['Sentiment'].values[test_indices]
    # Replace any -1 sentiment labels with 0
    test_labels[test_labels == -1] = 0

    # Prepare the BERT NLP model tokenizer to encode tweets. It is a crucial component for converting raw textual data
    # into a format that can be understood and processed by the BERT.

    # This line initializes the BERT tokenizer and sets its configuration,
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Use this to determine max length for encoding, (48)
    """
    encoded_texts = [tokenizer.encode(sent, add_special_tokens=True) for sent in train_texts]
    max_length = max([len(sent) for sent in encoded_texts])
    print('Max length: ', max_length)
    """

    # Encode TRAINING dataset for BERT.

    # BERT requires input sequences to be converted into numerical representations. Each token in the sequence is mapped
    # to a unique index in the model's vocabulary.
    train_inputs = []

    #BERT employs attention mechanisms to focus on specific parts of the input sequence. The attention mask indicates
    #which tokens the model should pay attention to and which ones to ignore.
    train_masks = []

    # For each tweet (train)
    for text in train_texts:
        # Encode the data. Return input encoding and attention mask

        # Tokenization is the process of breaking down the input text into smaller units (tokens).
        # Here, the tokenizer.encode_plus tokenizes the input text (text) using the previously initialized tokenizer.
        encoding = tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            padding='max_length',
            max_length=48,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True
        )

        # Add the encodings to the list
        train_inputs.append(encoding.get('input_ids'))
        train_masks.append(encoding.get('attention_mask'))

    #Return the lists as tensors

    # In PyTorch, a tensor is a fundamental data structure representing multi-dimensional arrays. Tensors can be scalars,
    # vectors, matrices, or even higher-dimensional arrays.

    # Neural networks, including models like BERT, are designed to operate on tensor data. Tensors provide a convenient
    # and efficient way to represent and manipulate numerical data, making them suitable for deep learning tasks.
    train_inputs = torch.cat(train_inputs)
    train_masks = torch.cat(train_masks)

    # Encode TESTING dataset for BERT
    test_inputs = []
    test_masks = []

    # For each tweet (test)
    for text in test_texts:
        # Encode the data. Return input encoding and attention mask

        # Tokenization and encoding process similar to the training set
        encoding = tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            padding='max_length',
            max_length=48,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True
        )

        # Add the encodings to the list
        test_inputs.append(encoding.get('input_ids'))
        test_masks.append(encoding.get('attention_mask'))

    # Return the lists as tensors
    test_inputs = torch.cat(test_inputs)
    test_masks = torch.cat(test_masks)

    # Get the train and test labels
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)

    #  Set the number of training examples utilized in one iteration. The model's weights are updated once per batch.
    batch_size = 16

    # Randomize the train data and define dataloader for model TRAINING.

    # TensorDataset allows you to create a dataset from a sequence of tensors.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    # SequentialSampler generates random indices for sampling elements from a dataset.
    train_sampler = RandomSampler(train_data)
    # DataLoader is an utility that creates an iterable over a dataset for iterating through batches.
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Randomize the test data and define dataloader for model TESTING

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    # Set random seed for repeatability. This don't directly generate random numbers, they ensure that the sequence of
    # random numbers generated by these libraries is the same every time the code is executed.
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    # Check if GPU is available and assign device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Initialize BERT Classifier
    model = BertClassifier(freeze=False)

    # Send model to device (GPU if available)
    model.to(device)

    # Define model hyperparameters

    # Specifies the number of times the entire training dataset is processed by the model during training
    epochs = 4
    # Determines the total number of iterations the model will undergo for weight updates
    steps = len(train_dataloader) * epochs
    # Controls the step size during optimization, influencing the size of the weight updates
    learning_rate = 5e-5
    # Small constant added to the denominator of some expressions to prevent division by zero
    epsilon = 1e-8
    # Initializes the optimizer, which is responsible for updating the model's weights during training
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
    # Control how the learning rate changes over time.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=steps)
    # Calculate the difference between the model's predictions and the actual labels during training.
    loss_function = nn.CrossEntropyLoss()

    print("BERT Model Accuracy:")

    # For the number of epochs
    for epoch in range(epochs):
        # Assign model to train
        model.train()

        # Initialize loss to zero
        train_loss = 0

        # For each TRAINING batch
        for batch in train_dataloader:
            # Get batch inputs, masks, and labels
            batch_inputs, batch_masks, batch_labels = batch

            # Send variables to device (GPU if available)
            batch_inputs = batch_inputs.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)

            # Reset the model gradient
            model.zero_grad()

            # Get classification of encoded values
            logits = model(batch_inputs, batch_masks)

            # Calculate loss based on predictions and known values
            loss = loss_function(logits, batch_labels)

            # Add loss to the running total
            train_loss += loss.item()

            # Update the model weights based on the loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Calculate the average loss over the batch
        train_loss /= len(train_dataloader)

        # Assign the model to evaluate
        model.eval()

        # Initialize losses
        test_loss = 0
        test_accuracy = 0

        # For each TESTING batch
        for batch in test_dataloader:
            # Get encoding inputs, masks, and labels
            batch_inputs, batch_masks, batch_labels = batch

            # Send variables to device (GPU if available)
            batch_inputs = batch_inputs.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)

            # Predict the input values without updating the model
            with torch.no_grad():
                logits = model(batch_inputs, batch_masks)

            # Calculate the loss
            loss = loss_function(logits, batch_labels)
            test_loss += loss.item()

            # Convert predictions to 0 and 1
            predictions = torch.argmax(logits, dim=1).flatten()

            # Calculate accuracy of the model on test data
            accuracy = (predictions == batch_labels).cpu().numpy().mean() * 100
            test_accuracy += accuracy

        # Calculate average loss and accuracy per each batch
        test_loss /= len(test_dataloader)
        test_accuracy /= len(test_dataloader)

        # Print epoch information
        print('\t Epoch: %d  |  Train Loss: %1.5f  |  Test Loss: %1.5f  |  Test Accuracy: %1.2f' % (
            epoch + 1, train_loss, test_loss, test_accuracy))

    # Save model
    torch.save(model.state_dict(), '../outputs/stock_sentiment_model.pt')


















