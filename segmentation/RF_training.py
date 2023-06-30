import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle
import time

categories = ['cropland', 'open-water','shrub', 'non-woody', 'wooded', 'other']
flat_data = []
target = []
start_time = time.time()

model = models.resnet18(pretrained=True)
layer = model._modules.get('avgpool')
model.eval()

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

for i in categories:
    path = os.path.join('segmentation/training/', i)
    for img in os.listdir(path):
        img = Image.open(os.path.join(path,img))
        tensor = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
        embedding = torch.zeros(512)
        def copy_data(m, i, o):
            embedding.copy_(o.data.reshape(o.data.size(1)))
        h = layer.register_forward_hook(copy_data)
        model(tensor)
        h.remove()
        flat_data.append(embedding)
        target.append(categories.index(i))
x = pd.DataFrame(flat_data)
y = np.array(target)

model = RandomForestClassifier()

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
kf = KFold(n_splits=5, shuffle=True)
for train_idx, test_idx in kf.split(x):
    x_train, x_test, y_train, y_test = x.iloc[train_idx], x.iloc[test_idx], y[train_idx], y[test_idx]
    model.fit(x_train, y_train)
#pickle.dump(model, open('segmentation/saved_models/06232023.sav', 'wb'))

    y_pred = model.predict(x_test)
    y_pred2 = model.predict(x_train)
    print(accuracy_score(y_train, y_pred2))
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred,pos_label='positive',average='micro'))
    recall_scores.append(recall_score(y_test, y_pred,pos_label='positive',average='micro'))
    f1_scores.append(f1_score(y_test, y_pred,pos_label='positive',average='micro'))
    cm = confusion_matrix(y_test, y_pred, labels=range(len(categories)))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
    display.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

end_time = time.time()
print(f"Test data: {np.mean(accuracy_scores)*100}% accurate, {np.mean(precision_scores)*100}% precision, {np.mean(recall_scores)*100}% recall, {np.mean(f1_scores)} f1 score")
print('Total Execution time:', end_time - start_time, 'seconds')