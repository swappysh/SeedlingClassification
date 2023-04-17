#!/usr/bin/env python
# coding: utf-8

# Load the data
from torchvision import datasets, transforms
from transformers import AutoImageProcessor
import torch, random, numpy as np

# setting random seed
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
size = (
    (image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)

# How many transformations are good?
transforms_resnet = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.Resize(size, antialias=None),
    # RandomResizedCrop being used here --> https://huggingface.co/docs/transformers/main/en/tasks/image_classification
    transforms.RandomRotation(360),
    transforms.RandomResizedCrop(size, antialias=None),
    transforms.ColorJitter(),
    transforms.RandomGrayscale(),
    transforms.RandomInvert(),
    # transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
])

# ### Sampling imbalance classes

# In[5]:


from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

def sampler(indices):
    labels = [dataset.targets[x] for x in indices]
    print(f'label length: {len(labels)}')
    distribution = dict(Counter(labels))
    class_weights = {k: 1/v for k, v in distribution.items()}

    samples_weight = np.array([class_weights[t] for t in labels])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


# In[6]:


from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
from collections import Counter

# Split validation data from training data
dataset = datasets.ImageFolder('./train', 
                               transform=transforms.Compose([
                                   transforms.Resize((256, 256), antialias=None),
                                   transforms.ToTensor()
                                   ]))

dataset_size = len(dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices) # shuffle the dataset before splitting into train and val
print(f'dataset_size: {dataset_size}')

split = int(np.floor(0.8 * dataset_size))
train_indices, val_indices = indices[:split], indices[split:]

# 
BATCH_SIZE = 24

train = DataLoader(Subset(dataset, train_indices), sampler=sampler(train_indices), batch_size=BATCH_SIZE)
val = DataLoader(Subset(dataset, val_indices), sampler=sampler(val_indices), batch_size=BATCH_SIZE)


# ### FineTuning resnet-50

# In[13]:


import torch

device = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if torch.backends.mps.is_built() else 
                      'cpu')


# In[16]:


from transformers import ResNetModel, ResNetConfig
from torch import nn
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

class CustomResNet(nn.Module):
    def __init__(self, checkpoint="microsoft/resnet-50", num_classes=12):
        super(CustomResNet, self).__init__()
        self.num_classes = num_classes
        self.model = ResNetModel.from_pretrained(checkpoint)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(2048, num_classes)
    
    def forward(self, x, labels=None):
        x = transforms_resnet(x)
        x = self.model(x)
        x = self.pooling(x[0])
        x = self.flatten(x)
        x = self.dropout(x)
        logits = self.classifier(x.view(-1, 2048))
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        
        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits)

model_ResNet = CustomResNet().to(device)


# In[14]:


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# In[15]:


from tqdm import tqdm

# Evaluation loop
def eval_loop(model, val, loss_fn, device, model_name="ensemble"):
    pred_cm, label_cm = torch.empty(0), torch.empty(0)
    total_loss, total_correct = 0, 0
    loops = 0
    model.eval()
    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(val)):
            image = image.to(device)
            label = label.to(device)
            
            output = model(image, labels=label)
            match model_name:
                case "resnet":
                    output = output.logits
                case "vit":
                    if label is not None:
                        output = output[1]
                    else:
                        output = output[0]
                        
            loss = loss_fn(output, label)
            
            total_loss += loss.item()
            loops += 1
            predicted = output.argmax(-1)
            total_correct += (predicted == label).sum().item()
            
            # store predicted and label for confusion matrix
            pred_cm = torch.cat((pred_cm, predicted.cpu()), 0)
            label_cm = torch.cat((label_cm, label.cpu()), 0)
            
        print(f'Validation Loss: {total_loss/loops:.2f}, Validation Accuracy: {(total_correct/(loops*BATCH_SIZE))*100:.2f}%')
        return total_loss/loops, (total_correct/(loops*BATCH_SIZE))*100, pred_cm, label_cm

# define trainingloop
def train_loop(model, train, val, optimizer, loss_fn, scheduler, early_stopper, epochs=10):
    new_lr = 0.1
    pred_cm, label_cm = torch.empty(0), torch.empty(0)
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        loops = 0
        for i, (image, label) in enumerate(tqdm(train)):
            image = image.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            output = model(image, labels=label)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loops += 1
            predicted = output.argmax(-1)
            total_correct += (predicted == label).sum().item()
        
        print(f'Epoch: {epoch}, Training Loss: {total_loss/loops:.2f}, Training Accuracy: {(total_correct/(loops*BATCH_SIZE))*100:.2f}%, Learning rate: {new_lr}')
        
        val_loss, val_acc, pred, label= eval_loop(model, val, loss_fn, device)
        pred_cm = torch.cat((pred_cm, pred), 0)
        label_cm = torch.cat((label_cm, label), 0)
        
        # Save model if validation accuracy is better than previous best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            try:
                torch.save(model.state_dict(), SAVE_MODEL)
            except NameError:
                torch.save(model.state_dict(), 'best_model.pt')
            print(f'Best model saved with validation accuracy: {best_val_acc:.2f}% and learning rate: {new_lr}')
        scheduler.step(total_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if early_stopper.early_stop(total_loss):             
            break
    
    return model, pred_cm, label_cm


# ## Train a ViT

# In[17]:


# Folder structure
# Training data
# contains images in 12 folders, each folder contains images of a single class
# Test data
# contains all images in a single folder

# Load the data
from torchvision import datasets, transforms
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
size = (
    (image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)

# How many transformations are good?
transforms_ViT = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.Resize(size, antialias=None),
    # RandomResizedCrop being used here --> https://huggingface.co/docs/transformers/main/en/tasks/image_classification
    transforms.RandomRotation(360),
    transforms.RandomResizedCrop(size, antialias=None),
    transforms.ColorJitter(),
    transforms.RandomGrayscale(),
    transforms.RandomInvert(),
    # transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
])

# In[19]:


from transformers import ViTModel
from torch import nn
from transformers.modeling_outputs import ImageClassifierOutput

class CustomViT(nn.Module):
    def __init__(self, checkpoint="google/vit-base-patch16-224", num_classes=12):
        super(CustomViT, self).__init__()
        self.num_classes = num_classes
        self.model = ViTModel.from_pretrained(checkpoint)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        # self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        # (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        # (classifier): Linear(in_features=768, out_features=1000, bias=True)
        
        self.classifier = nn.Linear(768, self.num_classes)
    
    def forward(self, 
                x,
                head_mask = None,
                labels = None,
                output_attentions = None,
                output_hidden_states = None,
                interpolate_pos_encoding = None,
                return_dict = None
                ):
        x = transforms_ViT(x)
        outputs = self.model(
            x,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

model_ViT = CustomViT().to(device)


# ## Ensemble of CustomResnet and another CustomResnet

# In[ ]:


class Ensemble(nn.Module):
    def __init__(self, model1, model2, num_classes=12):
        super(Ensemble, self).__init__()
        # Assuming model1 to be resnet and model2 to be ViT
        self.model1 = model1
        self.model2 = model2
        self.fc = nn.Linear(2*num_classes, num_classes, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # initialize weights
        identity = torch.eye(self.fc.weight.shape[0]//2, self.fc.weight.shape[1])
        self.fc.weight.data = torch.cat((identity, identity), dim=0)
        
    def forward(self, x, labels=None):
        x1 = self.model1(x, labels=labels)
        x2 = self.model2(x, labels=labels)
        
        if labels is not None:
            x2 = x2[1]
        else:
            x2 = x2[0]
        
        x = x1.logits
        x = torch.cat((x1.logits, x2), dim=1)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc(x)
        return x

# Load the models
model_ResNet.load_state_dict(torch.load("best_model.pt.8_apr_93", map_location=torch.device(device)))
model_ViT.load_state_dict(torch.load("best_model.pt.16_apr_ViT.80", map_location=torch.device(device)))

# Set the hyperparameters
from torch.optim import lr_scheduler

model_ensemble = Ensemble(model_ResNet, model_ViT).to(device)

# Freeze the weights of the models
for param in model_ensemble.model1.parameters():
    param.requires_grad = False
    
for param in model_ensemble.model2.parameters():
    param.requires_grad = False

count = 0
for param in model_ensemble.parameters():
    if param.requires_grad:
        count += 1
print(f'Number of trainable parameters: {count}')

epoch = 100
optimizer = torch.optim.SGD(model_ensemble.parameters(), lr=0.1)
criteria = torch.nn.CrossEntropyLoss()
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
early_stopper = EarlyStopper(patience=15, min_delta=0.001)

# evaluation on resnet
# eval_loop(model_ensemble, val, criteria, device)
# eval_loop(model_ResNet, val, criteria, device, model_name="resnet")
# exit()

# Train the ensemble model
model_ensemble, pred_cm, label_cm = train_loop(model_ensemble, train, val, optimizer, criteria, 
                                      scheduler, early_stopper, epochs=epoch)

# # Print the confusion matrix
from sklearn.metrics import confusion_matrix

# Confusion matrix
conf_mat=confusion_matrix(pred_cm.numpy(), label_cm.numpy())
print(conf_mat)

# Per-class accuracy
class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
print(class_accuracy)

# ## Saving the best model to file

# In[30]:


import os, glob
from PIL import Image
import pandas as pd
from torchvision import transforms

transforms = transforms.Compose([
    transforms.Resize((256, 256), antialias=None),
    transforms.ToTensor(),
])

# create empty dataframe
df = pd.DataFrame(columns=['file', 'species'])

# Run model over test data
for file_name in tqdm(glob.glob(os.path.join('./test', '*.png'))):
    image = transforms(Image.open(file_name)).to(device)
    output = model_ensemble(image.unsqueeze(0))
    predicted = output.argmax(-1).item()
    
    # concat to dataframe
    df = pd.concat([df, pd.DataFrame([{ 'file': file_name.split('/')[-1], 'species': dataset.classes[predicted] }])])

# Save file to csv
df.to_csv('submission.csv', index=False)
