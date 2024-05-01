# Style transfer data

For content images we pull from the COCO 2017 dataset (**INSERT CITATION**) which consists of labeled photographs of various objects and people. We use the API to filter only on photos that contain people, which leaves us with ~64k images to create a training set and ~3k images to create a validation set for our downstream classification.

For style images we pull from the Painter-by-Numbers dataset (**INSERT CITATION**), which consists of ~72k scans of actual paintings spanning dozens of artistic movements. 

For inference, we pair each content image with a random style image. These pairs are consistent across models in order to accurately compare the results of people detection.