import torch
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from torch.cuda.amp import autocast
import numpy as np
def evaluate_classification(model, test_loader):
    model.eval()
    y_true = torch.tensor([], dtype=torch.long).cuda()
    pred_probs = torch.tensor([]).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    # deactivate autograd engine and reduce memory usage and speed up computations
    with torch.no_grad():
        running_loss = 0.0
        for i, (X, y) in enumerate(test_loader):
            inputs = X.cuda()
            labels = y.cuda()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            y_true = torch.cat((y_true, labels), 0)
            pred_probs = torch.cat((pred_probs, outputs), 0)
            

    # compute predictions form probs
    y_true = y_true.cpu().numpy()
    _, y_pred = torch.max(pred_probs, 1)

    y_pred = y_pred.cpu().numpy()
    pred_probs = torch.nn.functional.softmax(pred_probs, dim=1).cpu().numpy()
    # get classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    # macro auc score
    # report["auc"] = roc_auc_score( y_true, pred_probs, multi_class="ovo", average="macro")
    # report["loss"] = running_loss / len(test_loader)
    print("Confusion Matrix: ")
    print(confusion_matrix(y_true, y_pred))

    return report