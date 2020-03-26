import time
import torch
from torch.autograd import Variable
import json

# data categories
data_cat = ['train', 'valid']

# Open JSON file
with open('Settings.json', 'r') as f:
    settings = json.load(f)

# Training and testing function
def train_model(model, criterion, optimizer, dataloaders,scheduler,
                dataset_sizes, num_epochs, costs, accs, num_ID, model_type):

    # Keeping track of time of each epoch
    since = time.time()

    best_acc = 0.0
    # Repeat training and testing for specified number of times
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in data_cat:
            model.train(phase == 'train')
            running_loss = 0.0
            running_corrects = 0

            # Loop through each batch of the dataset
            for i, data in enumerate(dataloaders[phase]):
                # Loop through each study of the batch
                for j, study in enumerate(data[0]):
                    inputs = study
                    labels = data[1][j].type(torch.Tensor)

                    # If in testing phase, do not change model
                    if phase == 'valid':
                        with torch.no_grad():
                            inputs = Variable(inputs.cuda())
                            labels = Variable(labels.cuda())
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            outputs = torch.mean(outputs)

                    else:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        outputs = torch.mean(outputs)

                    # Find loss and keep track of total loss
                    loss = criterion(outputs, labels, phase)
                    running_loss += loss.item()

                    # Check prediction and keep track of accuracy
                    preds = (outputs > 0.5).type(torch.cuda.FloatTensor)
                    running_corrects += torch.sum(torch.eq(preds, labels.data))

                # Change model if in training
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # Calculate the loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]

            # Keep track of acc and loss over epochs
            costs[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)

            # Print the loss & Accuracy for each epoch
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # copy the model
            if phase == 'valid':
                #Change step size according to loss
                scheduler.step(epoch_loss)
                # If the accuracy is the best so far, keep a copy of the model
                if epoch_acc == max(accs[phase]):
                    print('best model saved')
                    best_model_path = 'models/best_model_' + model_type + '.pth'
                    torch.save(model.state_dict(),best_model_path)
                    settings['run'][num_ID]['best_model_path'] = best_model_path

        # Determine the time it has taken to run the epoch
        time_elapsed = time.time() - since
        print('Time elapsed: {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

        # Saving parameters needed after each epoch incase we need to start again
        # Pytorch automatically converts the model weights into a pickle file
        latest_model_path = 'models/latest_model_' + model_type + str(num_ID) + '.pth'
        torch.save(model.state_dict(), latest_model_path)

        # costs,accs, lr will be auto updated from train
        settings['run'][num_ID]['costs'] = costs
        settings['run'][num_ID]['accuracy'] = accs
        settings['run'][num_ID]['lr'] = optimizer.param_groups[0]['lr']
        # saving model path for particular run
        settings['run'][num_ID]['latest_model_path'] = latest_model_path
        settings['run'][num_ID]['current_epoch'] = settings['run'][num_ID]['current_epoch'] + 1

        # store new accs and costs to JSON
        with open('Settings.json', 'w') as f:
            json.dump(settings, f, indent=2)

    # All the epochs have completed (training phase complete) -> This is how long it took the training phase to complete
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Print the best accuracy
    print('Best valid Acc: {:4f}'.format(best_acc))

    # load best model weights
    return model

# Testing the ensemble model
def test_ensemble_mean(model, criterion, dataloaders,dataset_sizes):
    running_loss = 0
    running_corrects = 0
    phase = "valid"

    for i, data in enumerate(dataloaders[phase]):
        for j, study in enumerate(data[0]):
            inputs = study
            labels = data[1][j].type(torch.Tensor)
            with torch.no_grad():
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # obtain outputs from model and take average
                pred1, pred2, pred3 = model(inputs)
                pred1 = torch.mean(pred1)
                pred2 = torch.mean(pred2)
                pred3 = torch.mean(pred3)
                pred_list = [pred1,pred2,pred3]
                final_pred = torch.mean(torch.stack(pred_list))

                loss = criterion(final_pred, labels, phase)
                running_loss += loss.item()

        # check prediction and keep track of accuracy
        preds = (final_pred > 0.5).type(torch.cuda.FloatTensor)
        running_corrects += torch.sum(torch.eq(preds, labels.data))


    # Calculate the loss and accuracy
    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc = running_corrects.item() / dataset_sizes[phase]

    return epoch_acc, epoch_loss
