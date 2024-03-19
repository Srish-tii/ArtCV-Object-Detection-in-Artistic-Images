# ArtCV-Object-Detection-in-Artistic-Images


## Steps to build StyleCOCO

StyleCOCO is generated from two image sets. `COCO 2017` contains the content images and `Painter By Numbers` contains the style images.

**COCO 2017**: On the linux server, `wget http://images.cocodataset.org/zips/train2017.zip` and unzip into `content_dir`

**Painter By Numbers**: This requires the Kaggle API. 

- Create a Kaggle account. 
- Go to `https://www.kaggle.com/settings`
- Go to API section, click `Create New Token`. Follow instructions provided to place JSON file in appropriate location on linux server
- Join Painter By Numbers competition: `https://www.kaggle.com/c/painter-by-numbers/`
- Run `kaggle competitions download -c painter-by-numbers -f train.zip -p /path/to/style_dir`


**Create StyleCOCO**: From the root project folder, `cd models/stylize` and run `nohup python3 stylize.py --content-dir '../../data/content_dir/' --style-dir '../../data/style_dir/' --output-dir '../../output/styled_images' --num-styles 1 --alpha 1 --content-size 0 --style-size 0 --crop 0 &`

