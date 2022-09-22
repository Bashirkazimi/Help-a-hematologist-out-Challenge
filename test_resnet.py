import os

import pandas as pd
import torchvision.models
from tqdm import tqdm

# import data_tanh


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

    pred_loader = data_fake_source_fancy_pca.get_test_phase_data('Datasets/WBC2/DATA-TEST', batch_size=1)



    experiment_name = 'resnet_test'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 11
    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model = torch.nn.DataParallel(model)
    model.to(device)

    model_load_path = 'models/resnet_train/model.pt'
    os.makedirs('models/{}'.format(experiment_name), exist_ok=True)
    os.makedirs('results/{}'.format(experiment_name), exist_ok=True)

    # loading the model with the highest validation accuracy
    print('loading model from: {}'.format(model_load_path))
    model.load_state_dict(torch.load(model_load_path))

    # Making prediction with the best model

    model.eval()
    image_names, labels, labelIDs = [], [], []
    print('making predictions on {} images'.format(len(pred_loader)))
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

    print('done!')

