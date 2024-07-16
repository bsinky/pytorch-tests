CIFAR tests
---

Based on [this tutorial here](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).

### train_images.py

This is the script to train a model. It will auto-download the CIFAR10 dataset, then train a simple network to classify the images into the 10 different classes represented in the dataset.

You can adjust the learning rate when running the script using the `-lr` argument.

```sh
python train_images.py -lr 0.0002 cifar-lr-0002.pth
```

After training finishes, the training script will automatically run the model against the test data and display the overall accuracy of the trained model.

### test_images.py

In addition, you can run a trained model against the test data again by running this script. The `-v` flag will also print out the accuracy for each individual class, to evaluate if the model(s) you've trained are getting better or worse at identifying specific subjects.

```sh
python test_images.py -v cifar-lr-0002.pth
```

