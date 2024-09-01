# cifar-10
Develop a model from scratch, without using any pre-existing
models or frameworks, and train it on the CIFAR-10 dataset. Submit the precision
and recall metrics for the trained model.  

**model.py**
Training a model without using any frameworks.
Was unable to get good accuracy
Got an accuracy of about 13.5%  

**model-kears.py**
trained the cifar-10 dataset using keras.  

saved the model as cifar10_model.h5  

`              precision    recall  f1-score   support

           0       0.83      0.73      0.78      1000
           1       0.82      0.91      0.86      1000
           2       0.75      0.53      0.62      1000
           3       0.53      0.66      0.59      1000
           4       0.73      0.71      0.72      1000
           5       0.71      0.61      0.66      1000
           6       0.72      0.89      0.79      1000
           7       0.83      0.77      0.80      1000
           8       0.86      0.85      0.86      1000
           9       0.79      0.87      0.83      1000

    accuracy                           0.75     10000
   macro avg       0.76      0.75      0.75     10000
weighted avg       0.76      0.75      0.75     10000
`
