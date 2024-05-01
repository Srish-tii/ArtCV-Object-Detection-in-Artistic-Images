# Style transfer demo

A few examples are included in the GitHub repository to facilitate running style transfer with all 3 models to generate the training examples for people detection.

**Preliminaries**

First make sure to `pip install -r requirements.txt`. Next, install the pre-trained weights for the various models [here](https://drive.google.com/drive/folders/1QVw2h3XUGjWaLZkFttwSCEjWR_EbtKZ9?usp=drive_link).

1. Within each `ArtFlow-X` folder is a `glow.pth`. Place each in the corresponding folder located in `ArtCV-Object-Detection-in-Artistic-Images/models/stylize/ArtFlow/experiments`.
2. Move the _contents_ of the `SANet` folder into `ArtCV-Object-Detection-in-Artistic-Images/models/stylize/SANet` (do not move entire folder).

**Inference with AdaIN/WCT**

Starting from the parent directory of the repository, `cd` to `models/stylize/ArtFlow` and run the following commmands.

For AdaIN:

`python3 test.py --content_dir ../../../data/examples/content --style_dir ../../../data/examples/style --decoder experiments/ArtFlow-AdaIN/glow.pth --output ../../../output/styled_images/examples/ArtFlow-AdaIn --operator adain --size 0 &`

For WCT:

`python3 test.py --content_dir ../../../data/examples/content --style_dir ../../../data/examples/style --decoder experiments/ArtFlow-WCT/glow.pth --output ../../../output/styled_images/examples/ArtFlow-WCT --operator wct --size 0 &`

**Inference with SANet**

Starting from the parent directory of the repository, `cd` to `models/stylize/SANet` and run the following commmand:

`python3 inference.py --content_dir ../../../data/examples/content --style_dir ../../../data/examples/style --vgg vgg_normalised.pth --output ../../../output/styled_images/examples/SANet --start_iter 500000 --size 0 &`