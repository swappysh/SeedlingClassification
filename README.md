# SeedlingClassification

- Using pre-trained ResNet-50.
- Using SGD instead of Adam. [Read somewhere that Adam works poorly on images (need to find citation)]
- Resizing and Normalizing images to standardize
- Yann was explaining how to set the batch size between [no_of_classes, 2*no_of_classes] in order to see every example of each class at least once.
- Using ReduceLROnPlateau which will reduce lr if validation loss doesn't decrease for 10 epochs. [Can use Adaptive Learning Rate Methods but as previously noted SGD seems to do better]
- Confusion matrix shows confusion in Class 0 and 6. Suggestions from kaggle discussion point to use train another model and combine using Ensemble methods.

Best models:
https://drive.google.com/file/d/11uwYAs0ccbN5_ZAfqnV8y6Us28yo7cCi/view?usp=sharing
https://drive.google.com/file/d/1Drn5T0_zpy_9gIDSqE6sn41YlRz9g7Fe/view?usp=sharing
