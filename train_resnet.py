import os

import pandas as pd
import torchvision.models
from tqdm import tqdm

# import data_tanh

import hrnet

import data_fake_source_fancy_pca

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix, matthews_corrcoef, classification_report,confusion_matrix, accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,  precision_score, recall_score

import matplotlib.pyplot as plt
import torch
torch.manual_seed(42)

import random
random.seed(0)

import numpy as np
np.random.seed(0)

label_map_reverse = {
        0: 'basophil',
        1: 'eosinophil',
        2: 'erythroblast',
        3: 'myeloblast',
        4: 'promyelocyte',
        5: 'myelocyte',
        6: 'metamyelocyte',
        7: 'neutrophil_banded',
        8: 'neutrophil_segmented',
        9: 'monocyte',
        10: 'lymphocyte_typical'
    }

label_map_all = {
        'basophil': 0,
        'eosinophil': 1,
        'erythroblast': 2,
        'myeloblast' : 3,
        'promyelocyte': 4,
        'myelocyte': 5,
        'metamyelocyte': 6,
        'neutrophil_banded': 7,
        'neutrophil_segmented': 8,
        'monocyte': 9,
        'lymphocyte_typical': 10
    }


def make_predictions(pred_loader, model, device):
    n = len(pred_loader)
    model.eval()
    preds = torch.tensor([], dtype=int)
    y_true = torch.tensor([], dtype=int)
    preds = preds.to(device)
    prediction = torch.tensor([])
    prediction = prediction.to(device)
    for i, data in enumerate(pred_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        x, y, _ = data
        x = x.to(device)
        out = model(x)
        logits = torch.softmax(out.detach(), dim=1)
        prediction = torch.cat((prediction, logits), 0)
        predic = logits.argmax(dim=1)
        preds = torch.cat((preds, predic), 0)
        y_true = torch.cat((y_true, y), 0)

    preds = preds.cpu()
    y_true = y_true.cpu().numpy()
    preds = preds.detach().numpy()

    y_true = [label_map_reverse[y] for y in y_true]
    preds = [label_map_reverse[p] for p in preds]
    return y_true, preds


def classification_complete_report(y_true, y_pred, labels=None, experiment_name='experiment'):
    print(classification_report(y_true, y_pred, labels=None))
    print(15 * "----")
    print("matthews correlation coeff: %.4f" % (matthews_corrcoef(y_true, y_pred)))
    print("Cohen Kappa score: %.4f" % (cohen_kappa_score(y_true, y_pred)))
    print("Accuracy: %.4f & balanced Accuracy: %.4f" % (
    accuracy_score(y_true, y_pred), balanced_accuracy_score(y_true, y_pred)))
    # print("macro F1 score: %.4f & micro F1 score: %.4f" % (f1_score(y_true, y_pred, average = "macro"), f1_score(y_true, y_pred, average = "micro")) )
    print("macro Precision score: %.4f & micro Precision score: %.4f" % (
    precision_score(y_true, y_pred, average="macro"), precision_score(y_true, y_pred, average="micro")))
    print("macro Recall score: %.4f & micro Recall score: %.4f" % (
    recall_score(y_true, y_pred, average="macro"), recall_score(y_true, y_pred, average="micro")))
    print(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 10))  # plot size
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax, include_values=False, colorbar=False)

    # plt.show()
    plt.savefig('results/{}/cm.png'.format(experiment_name))
    print(15 * "----")


if __name__ == '__main__':

    # get datasts
    fake_dir = 'Datasets/MAT_ACE_AS_WBC_MEAN_STD'
    train_loader, valid_loader, test_loader = data_fake_source_fancy_pca.get_train_val_test('metadata.csv', fake_dir=fake_dir)

    pred_loader = data_fake_source_fancy_pca.get_pred_loader_mean_std('metadata.csv', batch_size=1)

    experiment_name = 'resnet_train'
    epochs = 50
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 11
    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model = torch.nn.DataParallel(model)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model_save_path = 'models/{}/model.pt'.format(experiment_name)
    os.makedirs('models/{}'.format(experiment_name), exist_ok=True)
    os.makedirs('results/{}'.format(experiment_name), exist_ok=True)

    # running variables
    epoch = 0
    update_frequency = 5  # number of batches before viewed acc and loss get updated
    counter = 0  # counts batches
    f1_macro_best = 0  # minimum f1_macro_score of the validation set for the first model to be saved
    loss_running = 0
    acc_running = 0
    val_batches = 0

    y_pred = torch.tensor([], dtype=int)
    y_true = torch.tensor([], dtype=int)
    y_pred = y_pred.to(device)
    y_true = y_true.to(device)

    # Training

    for epoch in range(0, epochs):
        # training
        model.train()

        with tqdm(train_loader) as tepoch:
            for i, data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch + 1}")
                counter += 1

                x, y, _ = data
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()

                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                logits = torch.softmax(out.detach(), dim=1)
                predictions = logits.argmax(dim=1)
                acc = accuracy_score(y.cpu(), predictions.cpu())

                if counter >= update_frequency:
                    tepoch.set_postfix(loss=loss.item(), accuracy=acc.item())
                    counter = 0

        # validation
        model.eval()
        with tqdm(valid_loader) as vepoch:
            for i, data in enumerate(vepoch):
                vepoch.set_description(f"Validation {epoch + 1}")

                x, y, _ = data
                x, y = x.to(device), y.to(device)

                out = model(x)
                loss = criterion(out, y)

                logits = torch.softmax(out.detach(), dim=1)
                predictions = logits.argmax(dim=1)
                y_pred = torch.cat((y_pred, predictions), 0)
                y_true = torch.cat((y_true, y), 0)

                acc = accuracy_score(y_true.cpu(), y_pred.cpu())

                loss_running += (loss.item() * len(y))
                acc_running += (acc.item() * len(y))
                val_batches += len(y)
                loss_mean = loss_running / val_batches
                acc_mean = acc_running / val_batches

                vepoch.set_postfix(loss=loss_mean, accuracy=acc_mean)

            f1_micro = f1_score(y_true.cpu(), y_pred.cpu(), average='micro')
            f1_macro = f1_score(y_true.cpu(), y_pred.cpu(), average='macro')
            print(f'f1_micro: {f1_micro}, f1_macro: {f1_macro}')
            if f1_macro > f1_macro_best:
                f1_macro_best = f1_macro
                torch.save(model.state_dict(), model_save_path)
                print('model saved at {}'.format(model_save_path))

            # reseting running variables
            loss_running = 0
            acc_running = 0
            val_batches = 0

            y_pred = torch.tensor([], dtype=int)
            y_true = torch.tensor([], dtype=int)
            y_pred = y_pred.to(device)
            y_true = y_true.to(device)

    print('Finished Training')

    # loading the model with the highest validation accuracy
    model.load_state_dict(torch.load(model_save_path))

    # evaluating on the test set
    print('Evaluate on the test set')
    y_true, y_pred = make_predictions(test_loader, model, device)
    classification_complete_report(y_true, y_pred, list(label_map_all.keys()), experiment_name=experiment_name)

    # Making prediction with the best model

    model.eval()
    image_names, labels, labelIDs = [], [], []

    for i, data in enumerate(pred_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        x, image_name = data
        x = x.to(device)
        out = model(x)
        logits = torch.softmax(out.detach(), dim=1)
        predic = logits.argmax(dim=1)
        labelID = predic.cpu().numpy()[0]
        labelIDs.append(labelID)
        labels.append(label_map_reverse[labelID])
        image_names.append(image_name[0])

    df = pd.DataFrame({'Image': image_names, 'Label': labels, 'LabelID': labelIDs})
    df.to_csv('results/{}/submission.csv'.format(experiment_name))
