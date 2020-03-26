import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from densenet import densenet169
from train import train_model, test_ensemble_mean
from datapipeline import get_study_data, get_dataloaders
from resnet import resnet101
from VGG import vgg16_bn
import json
from ensemble import Ensemble
from shufflenet import shufflenet_v2_x1_0

# Weighted cross entropy loss
class Loss(nn.modules.Module):
    def __init__(self, norm_weight, ab_weight):
        super(Loss, self).__init__()
        self.norm_weight = norm_weight
        self.ab_weight = ab_weight

    def forward(self, inputs, targets, phase):
        loss = - (self.norm_weight[phase] * targets * inputs.log() + self.ab_weight[phase] * (1 - targets) * (
                1 - inputs).log())
        return loss

# obtain count for pos or neg studies
def get_count(df, cat):
    return df[df['Path'].str.contains(cat)]['Count'].sum()

if __name__ == '__main__':

    # load JSON file
    with open('Settings.json', 'r') as f:
        settings = json.load(f)

    # selecting run in JSON file
    num_ID = 29

    # load variables from JSON file
    batch_size = settings['run'][num_ID]['bs']
    current_epoch = settings['run'][num_ID]['current_epoch']
    epochs = settings['run'][num_ID]['total_epochs']
    learning_rate = settings['run'][num_ID]['lr']
    droprate = settings['run'][num_ID]['dropout']
    costs = settings['run'][num_ID]['costs']
    accs = settings['run'][num_ID]['accuracy']
    latest_model_path = settings['run'][num_ID]['latest_model_path']
    model_type = settings['run'][num_ID]['model_type']

    # load study level data into batches
    study_data = get_study_data(study_type='XR_WRIST')
    data_cat = ['train', 'valid']  # data categories
    dataloaders = get_dataloaders(study_data, batch_size)
    dataset_sizes = {x: len(study_data[x]) for x in data_cat}

    # tai = total abnormal images, tni = total normal images
    tai = {x: get_count(study_data[x], 'positive') for x in data_cat}
    tni = {x: get_count(study_data[x], 'negative') for x in data_cat}

    # Find the weights of abnormal images and normal images
    Wt1 = {x: (tni[x] / (tni[x] + tai[x])) for x in data_cat}
    Wt0 = {x: (tai[x] / (tni[x] + tai[x])) for x in data_cat}

    # For training & testing individual models
    if model_type != 'ensemble':
        if model_type == "dense":
            model = densenet169(pretrained=True, droprate= droprate)

        elif model_type == "vgg":
            model = vgg16_bn(pretrained=True)
            model.classifier[6] = nn.Linear(4096, 1)

        elif model_type == "shufflenet":
            model = shufflenet_v2_x1_0()
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 1)

        else:
            model = resnet101()
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 1)

        if latest_model_path != "":
            model.load_state_dict(torch.load(latest_model_path))

        model.cuda()

        # Set parameters for model
        criterion = Loss(Wt1, Wt0)
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience = 1, verbose = True)

        # Train model
        model = train_model(model, criterion, optimizer, dataloaders, scheduler, dataset_sizes, epochs - current_epoch, costs, accs, num_ID, model_type)

    # For testing ensemble model
    else:
        model = Ensemble("models/best_model_dense_4.pth","models/best_model_res_12.pth","models/best_model_vgg_19.pth")
        model.cuda()
        criterion = Loss(Wt1, Wt0)
        test_acc,test_loss= test_ensemble_mean(model, criterion, dataloaders, dataset_sizes)