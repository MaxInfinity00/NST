##Able to style transfer among images of different dimensions :), need to resize post processing tho
import copy

import kornia
import torch
import colab_utils
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import models
from torchvision import transforms
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#setting device according to efficiency
print(device)

#Loading Images
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
content_img = colab_utils.load_image("style_reference/Shapes.jpg", size=imsize)
content_img = content_img.unsqueeze(0).to(device)
style_img = colab_utils.load_image("butterfly.jpg", size=imsize)
style_img = style_img.unsqueeze(0).to(device)

colab_utils.imshow(content_img, title="Content Image")
colab_utils.imshow(style_img, title="Style Image")

class ImageNetNormalize(nn.Module):
  """Module which normalizes inputs using the ImageNet mean and stddev."""

  def __init__(self):
    super().__init__()
    self._transform = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

  def forward(self, input):
    # Remove the batch dimension before transforming then add it back.
    # Warning: This only produces sane results when batch_size=1.
    return self._transform(input.squeeze(0)).unsqueeze(0)

#Computing Content Loss
class ContentLoss(nn.Module):
    """The content loss module.
    
    Computes the L1 loss between the target and the input.
    """
    def __init__(self, target,):
        """Initializes a new ContentLoss instance.
        Args:
            target: Take the L1 loss with respect to this target.
        """
        super().__init__()
        # we 'detach' the target content from computation graph since we want to use the actual value.
        self.target = target.detach()
        self.loss = None

    def forward(self, input):
        self.loss = F.l1_loss(input, self.target)
        return input

#Style Loss Computation
class StyleLoss(nn.Module):
    """The style loss module.
    
    Computes the L1 loss between the gram matricies of the target feature and the
    input.
    """
    def __init__(self, target_feature):
        """Initializes a new StyleLoss instance.
        Args:
            target_feature: Take the L1 loss with respect to this target feature.
        """
        super().__init__()
        # We detach the target_feature and target from the computation graph since we want to use the actual values.
        self._target = self._gram_matrix(target_feature.detach()).detach()
        self.loss = None
        
    def _gram_matrix(self, input):
        """Returns the normalized Gram matrix of the input."""
        a, b, c, d = input.size() # a=batch size(=1), b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        # we 'normalize' the values of the gram matrix by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)
    
    def forward(self, input):
        G = self._gram_matrix(input)
        self.loss = F.l1_loss(G, self._target)
        return input

# desired depth layers to compute style/content losses :
content_layers_default = ['relu4_2']
style_layers_default = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

def get_nst_model_and_losses(model, style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    """Creates the Neural Style Transfer model and losses. 
    
    We assume the model was pretrained on ImageNet and normalize all inputs using
    the ImageNet mean and stddev.
    
    Args:
      model: The model to use for Neural Style Transfer. ContentLoss and StyleLoss
        modules will be inserted after each layer in content_layers and 
        style_layers respectively.
      content_img: The content image to use when creating the ContentLosses.
      style_img: The style image to use when creating the StyleLosses.
      content_layers: The name of the layers after which a ContentLoss module will
        be inserted.
      style_layers: The name of the layers after which a StyleLoss module will be
        inserted.
    Returns: A three item tuple of the NST model with ContentLoss and StyleLoss 
      modules inserted, the ContentLosses modules, and the StyleLosses modules.
    """
    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    nst_model = nn.Sequential(ImageNetNormalize())
    cnn = copy.deepcopy(model)
    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses, style_losses, last_layer = [], [], 0

    for i, (name, layer) in enumerate(cnn.named_children()):
        nst_model.add_module(name, layer)
        
        if name in content_layers:
            # add content loss:
            content_loss = ContentLoss(nst_model(content_img))
            nst_model.add_module(f'{name}_ContentLoss', content_loss)
            content_losses.append(content_loss)
            last_layer = i

        if name in style_layers:
            # add style loss:
            style_loss = StyleLoss(nst_model(style_img))
            nst_model.add_module(f'{name}_StyleLoss', style_loss)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
            last_layer = i
    # Sanity check that we have the desired number of style and content layers.
    assert len(content_losses) == len(content_layers), 'Not all content layers found.'
    assert len(style_losses) == len(style_layers), 'Not all style layers found.'
    # Remove the layers after the last StyleLoss and ContentLoss since they will
    # not be used for style transfer. To get the correct last_layer index, we 
    # take into account the ImageNetNormalization layer at the front and the
    # ContentLoss and StyleLoss layers.
    last_layer += 1 + len(content_losses) + len(style_losses)
    nst_model = nst_model[:last_layer+1].to(device)
    return nst_model, style_losses, content_losses
    
def rename_vgg_layers(model):
  """Renames VGG model layers to match those in the paper."""
  block, number = 1, 1
  renamed = nn.Sequential()
  for layer in model.children():
    if isinstance(layer, nn.Conv2d):
      name = f'conv{block}_{number}'
    elif isinstance(layer, nn.ReLU):
      name = f'relu{block}_{number}'
      # The inplace ReLU version doesn't play nicely with NST.
      layer = nn.ReLU(inplace=False)
      number += 1
    elif isinstance(layer, nn.MaxPool2d):
      name = f'pool_{block}'
      # Average pooling was found to generate images of higher quality than
      # max pooling by Gatys et al.
      layer = nn.AvgPool2d(layer.kernel_size, layer.stride)
      block += 1
      number = 1
    else:
      raise RuntimeError(f'Unrecognized layer "{layer.__class__.__name__}""') 
    renamed.add_module(name, layer)
  return renamed.to(device)

model = models.vgg19(pretrained=True).features.to(device).eval()
model = rename_vgg_layers(model)

def run_style_transfer(model, content_img, style_img, content_layers=content_layers_default, style_layers=style_layers_default,
                       input_img=None, num_steps=512,content_weight=1, style_weight=1e9, log_steps=50):
    """Runs Neural Style Transfer.
    
    Args:
      model: The Neural Style Transfer model to use.
      content_image: The image whose content to match during the optimization. 
      style_image: The image whose style to match during the optimization.
      content_layers: The names of the layers whose output will be used to compute
        the content losses.
      style_layers: The names of the layers whose output will be used to compute
        the style losses.
      input_img: The image which will be optimized to match the content and style
        of the content_img and style_img respectively. If None, defaults to random
        Gaussian noise.
      num_steps: The number of steps to run the optimization for.
      content_weight: A weight to multiply the content loss by. 
      style_weight: A weight to multiply the style loss by. 
      log_steps: The number of consecutive training steps to run before logging.
    Returns:
      The optimized input_img.
    """ 
    n, c, h, w = content_img.data.size()
    if input_img is None:
      input_img = torch.randn((n, c, h, w), device=device)
      input_img = input_img * .01  # Scale the noise variance down.
    model, style_losses, content_losses = get_nst_model_and_losses(model, style_img, content_img,content_layers, style_layers)
    optimizer = optim.Adam([input_img.requires_grad_()], lr=.05)
    transform = nn.Sequential(
      kornia.augmentation.RandomResizedCrop(size=(w, h), scale=(.97, 1.), ratio=(.97, 1.03)),
      kornia.augmentation.RandomRotation(degrees=1.))
    for step in range(num_steps):
        optimizer.zero_grad()
        input_img.data.clamp_(0, 1)
        model(transform(input_img))
        style_score, content_score = 0, 0
        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss
            
        style_score *= style_weight
        content_score *= content_weight
        loss = style_score + content_score
        loss.backward()
        optimizer.step()
        if (step > 0 and step % log_steps == 0) or (step + 1) == num_steps:
              print(f'[{step}]: content_loss={content_score.item()},'
                    f' style_loss={style_score.item():4f}')
              #colab_utils.imshow(input_img.data.clamp_(0, 1), figsize=(10, 10))
        torch.cuda.empty_cache()
        #if(step % 10 == 0):
        if(step<=100 and step%5==0) or (step>=800 and step%40==0) or (100<step<800 and step%25==0):
            save_image(input_img,'test/iter'+''.join(['0' for i in range(len(str(num_steps)) - len(str(step)))])+str(step)+'.png')

    return input_img

#input_img = None
input_img = content_img.clone()
#input_img = style_img.clone()
# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)

output = run_style_transfer(model, content_img, style_img,
                            # we found that relu3_2 produces more visually appealing results.
                            content_layers=['relu3_2'], style_layers=style_layers_default, input_img=input_img,
                            num_steps=1024, content_weight=1., style_weight=1000., log_steps=250)
save_image(output, 'finalresult.png')
#save_image(output, 'witchatstake.png')
