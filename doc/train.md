# Train the model

We'll use PyTorch to train the ResNet18 model.

The `train_model_resnet18.py` program uses the images free/blocked that are located under the __dataset__ folder created a the previous data collection step.

> Note that the train on a CPU will take looong time, so prefer transfering the images on a hardware equiped with a NVidia GPU and perform train on that GPU.

 Run the program:

 ``` bash
python3 train_model_resnet18.py
```

The output is saved to a PyTorch [`state_dic`](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict)

> The output file is "__best_model_resnet18.pth__", get back this file to your LeoMower scripts folder.

> The pictures collected in my garden and trained with the ResNet18 model are 'saved' in ths state_dic `best_model_resnet18_garden.pth`

`best_model_resnet18_garden.pth` is the state_dic used for [the inference at the next step](infer.md)


## Some useful references

* https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5
* https://vitalflux.com/pytorch-load-predict-pretrained-resnet-model/
* https://github.com/cfotache/pytorch_imageclassifier/blob/master/PyTorch_Image_Inference.ipynb
