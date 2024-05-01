## Style transfer methodology

We built upon the work of Kadish et. al. (2021) by considering two alternative--and potentially improved--style transfer techniques to Adaptive Instance Normalization (AdaIN): Whitening and Coloring Transforms and Style-Attention Network. Using these two models, we will generate competing synthetic training sets for person detection and compare their accuracy relative to the baseline model trained with AdaIN-generated stylized photos.

### Whitening and color transforms (WCT)

WCT (Huang & Belongie, 2017) and AdaIN share a similar architecture (see diagram below). Both consist of a pre-trained VGG-19 encoder which maps the input content and style images into a high-dimensional feature space. Then, certain feature layers are linearly transformed and passed through the decoder which is learned in model training. The output of the decoder is the stylized content image.

<img src="images/adain_wct_arch.png"
     alt="AdaIN/WCT Architecture"
     style="width: 75%; height: auto"/>

Before contrasting the models, a quick note on the taxonomy of the feature space. The feature space consists of feature layers, where each layer may consist of multiple channels. Earlier feature layers tend to capture semantic patterns in a mapped image, whereas deeper layers will capture more localized patterns.

The models differ in the linear transformation that is applied to feature layers after encoding. For a given feature layer, AdaIN matches the channel-wise mean and variance of the content and style images by shifting and scaling the feature layer of the content image accordingly. WCT intends to perform a similar but improved operation. Instead of considering just channel-wise relationships, WCT incorporates inter-channel relationships by applying a two-fold linear transformation. First, a whitening transformation is applied to the content image's feature layer to attempt to strip the image of its stylistic properties. Next, a coloring transformation (which is a function of the style image's feature layer) is applied to the whitened content image feature layer to attempt to apply the styistic properties of the style image to the content image. Note that $f_{c}$ is a matrix of channels of the feature layer of the content image $c$ (i.e., $f_{c} \in \mathbb{R}^{C \times (h*w)}$, where $C$ is the number of channels and $h$ and $w$ are the height and width of the content image, respectively). Also note that the transformation matrix is (roughly) the eigendecomposition of $f_{c}f_{c}^{T}$, which when normalized is the covariance matrix of $f_{c}$. This further explains the notion that WCT is accounting for cross-channel relationships.

### Style-attention network (SANet)

The second model we experiment with is SANet (Park & Lee 2019).

<img src="images/sanet_arch.png"
     alt="SANet Architecture"
     style="width: 75%; height: auto"/>

The model broadly shares the same structure as AdaIN and WCT (see diagram above). The content and style images are mapped to the feature space with a pre-trained VGG-19 encoder. Select feature layers are transformed, upsampled and concatenated to then be sent through a learned decoder, which takes the input and produces the stylized content image.

Where SANet differs from the other two models is in the transform stage. Unlike WCT and AdaIN, SANet performs a nonlinear transformation of the feature layers. Specifically, the authors incorporate an attention mechanism to capture not just inter-channel relationships within a single feature layer, but inter-layer relationships across the feature space. The motivation behind this choice is that each feature layer captures different attributes of the images. The shallower feature layers tend to capture semantic relationships in an image, whereas the deeper layers typically capture more localized patterns. In attending to a given feature layer of the content image with every feature layer of the style image, the authors hope to identify semantically similar regions in the content and style images and apply the style of that region in the style image to the corresponding region in the content image. Mathematically, this is formulated as

$$F^{i}_{cs} = \frac{1}{C(F)}\sum_{\forall j}\text{exp}\left(f(\bar{F^{i}_{c}})^{T}g(\bar{F^{j}_{s}})\right)h(F^{j}_{s})$$

where $F^{i}_{cs}$ is the $i^{th}$ feature layer of the stylized content image, $\{f(\centerdot), g(\centerdot), h(\centerdot)\}$ are linear transformations whose matrix weights are learned, $\{c,s\}$ represent the content and style images (respectively), $C(F) =  \sum_{\forall j}\text{exp}\left(f(\bar{F^{i}_{c}})^{T}g(\bar{F^{j}_{s}})\right)$ and $\bar{F^{i,j}_{c,s}}$ is the mean-variance channel-wise normalized version of $F^{i,j}_{c,s}$. In essence this expression represents the weighted sum of the feature layer of the style image where the weights are determined by the similarity between the $i^{th}$ and $j^{th}$ feature layer of content and style images, respectively. This is then mapped to a matrix of values between 0 and 1 by wrapping it in the softmax function.

