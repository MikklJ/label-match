# Label & Match

## Introduction

The repo for matching annotations from distinct video frames. This research has been done by Michael Ji at Blink AI.

## Progress

- [x] Initialize repository with design goals (10 minutes)
- [X] Learn using gcloud resources and transition code to google cloud (1/2 day)
- [X] Learn using git(add&commit&pull&push pipeline) and transition code to bitbucket so two people can work on the project at the same time. (1/2 day)
- [ ] **Prepare Dataloader for getting cropped images by using the bboxes of bdd-dataset during training (let's call this dataset "Cropped")** (1 day)
- [ ] **Implement siamese or triplet networks** (3 days)
- [ ] **Prepare siamese loss (or triplet loss)** (1-2 days)
- [ ] **Training of siamese networks can be done online, but for best results train the network with considerable amount of cropped images** (3 days)
- [ ] **Formalize report of accuracies and validate against ground truth labels, you can use your previous labels for this** (1 day)
- [ ] **Integrate Siamese network into the annotation workflow** (1 day)
- [ ] **Demo Results** (1 day)
- [ ] (optional) visualize shallow model on the Cropped Database (1/2 day)



## Other thoughts and notes
* we will make a new directory automatically each time we run ```train``` which contains for that run
    * config
    * tensorboard log
    * saved models (the correct way https://pytorch.org/docs/master/notes/serialization.html?highlight=save)
* we will use the same data augmentation and image resize for all models. Learning rate can be model dependent.
* we will use git to keep each other up-to-date on implementation successes/failures

## Tools
* virtualenv
* python3, pytorch 1.1.0
* pytorch Imagenet pretrained backbones - https://github.com/Cadene/pretrained-models.pytorch
* google cloud for training - https://cloud.google.com
* tensorboardx for viz (browser is ideal) - https://github.com/lanpa/tensorboardX
* yaml for config - (example usage: https://github.com/NVlabs/MUNIT/blob/master/configs/demo_edges2handbags_folder.yaml)
