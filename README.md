# Optical Character Recognition on digits

This model aims at performing OCR on words composed of digits.
For now it can only take a digit word (as an image) as input.

If you want to train the model on N epochs then do
`python train_mnist.py --mode train --epochs N`

If you want to test it on the MNIST dataset do
`python train_mnist.py --mode test`

If you want to test it on custom MNIST dataset do
`python train_mnist.py --mode custom_mnist`

To run on a CUDA GPU, just add `--gpu X` where X is the GPU number you want to use.

Finally, to run the full model, just run
`python number_image_classifier.py`

Feel free to add other images in the folder data/NumberImages.