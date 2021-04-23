# pstage_02_image_classification


## code explaination
* EDA
  - EDA code.
* sub
  - inference with trained model and make sub.csv
* tsne
  - another EDA-like code. visualize logits of validation data set using t-sne.
* dataset
  - define DataSet.
* focal_loss
  - define Focal loss
* label_smoothing
  - define Label Smoothing loss
* avgMeter
  - calculate metric like accuracy or f1 score.
* acc_per_label
  - calculate accuracy for every class.
* effnet_b0_fine_tune
  - training efficientnet-b0 and save model weights with the best f1 score.
## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Install Requirements
- `pip install -r requirements.txt`
