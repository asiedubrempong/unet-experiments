#!/usr/bin/env python
# coding: utf-8

# # UNet

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from fastai import *
from fastai.vision import *


# # Data 

# check if pytorch is using gpu

# In[3]:


torch.cuda.current_device()


# In[4]:


torch.cuda.get_device_name(0)


# In[5]:


path = Path('../data/carvana/')
path.ls()


# In[6]:


fnames = get_image_files(path/'train')
fnames[:3]


# In[7]:


labels = get_image_files(path/'train_masks')
labels[:3]


# In[8]:


fname = fnames[0]
img = open_image(fname)
img.show(figsize=(5,5))


# function to return the path to the mask for any training image

# In[9]:


def get_mask(fname): return Path(str(fname.parent) + '_masks') / (fname.name[:-4] + '_mask.gif')


# In[10]:


mask = open_mask(get_mask(fname))
mask.show(figsize=(5,5), alpha=1)


# In[11]:


src_size = np.array(mask.shape[1:])
src_size, mask.data


# ## Create a Dataset

# In[12]:


codes = array(['background', 'car'])


# In[13]:


bs, size = 4, 512
bs, size


# **change SegmentationItemList class to work on binary segmentation**

# In[14]:


class SegLabelListCustom(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True)
    
class SegItemListCustom(ImageItemList):
    _label_cls = SegLabelListCustom


# In[15]:


src = (SegItemListCustom.from_folder(path/'train')
       .random_split_by_pct()
       .label_from_func(get_mask, classes=codes))


# In[16]:


data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))


# In[17]:


data.show_batch(2, figsize=(10,7))


# ## Model Architecture

# In[18]:


def unet_block(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU())


# In[19]:


def upsample_block(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)


# In[20]:


def center_crop(features, target_size_h, target_size_w):
    """
        When combining the upsampled output with the output of the encoder, the features 
        from the encoder have a larger height and width than the upsampled output
       
        The features from the contracting path are cropped to have the same size
        as the upsampled output.
    """
    _, _, encoder_size_h, encoder_size_w = features.shape
    crop_size_h = (encoder_size_h - target_size_h) // 2
    crop_size_w = (encoder_size_w - target_size_w) // 2
    
    return features[:, :, crop_size_h:(crop_size_h + target_size_h), 
                   crop_size_w:(crop_size_w + target_size_w)]


# In[21]:


class DecoderBlock(nn.Module):
    """
        Implements one step of the expansive path
        
        Every step in the expansive path consists of:
        1.An upsampling of the feature map followed by a 2x2 convolution 
          that halves the number of feature channels.
        2.A concatenation with the correspondingly cropped feature map from the
          contracting path, and two 3x3 convolutions, each followed by a relu.
        
    """
    def __init__(self, n_channels):
        super().__init__()
        
        self.up_conv = upsample_block(n_channels, n_channels//2)    
        self.normal_conv = unet_block(n_channels, n_channels//2)
    
    def forward(self, x, hook):
        out = self.up_conv(x)
        out = torch.cat((center_crop(hook, out.shape[2], out.shape[3]),
                            out), 1)
        out = self.normal_conv(out)
        return out


# In[22]:


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        ### contracting path
        self.c_one = unet_block(in_channels, 64)  # first step in contracting path
        self.c_two = unet_block(64, 128)
        self.c_three = unet_block(128, 256)
        self.c_four = unet_block(256, 512)
        self.c_five = unet_block(512, 1024)
        
        self.max_pool = nn.MaxPool2d(2, stride=2)
        
        ### expansive path
        self.e_one = DecoderBlock(1024)
        self.e_two = DecoderBlock(512)
        self.e_three = DecoderBlock(256)
        self.e_four = DecoderBlock(128)
        
        self.final_layer = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        hook_one = self.c_one(x)
        hook_two = self.c_two(self.max_pool(hook_one))
        hook_three = self.c_three(self.max_pool(hook_two))
        hook_four = self.c_four(self.max_pool(hook_three))
        
        encoder_output = self.c_five(self.max_pool(hook_four))
        
        e_one = self.e_one(encoder_output, hook_four)
        e_two = self.e_two(e_one, hook_three)
        e_three = self.e_three(e_two, hook_two)
        e_four = self.e_four(e_three, hook_one)
        
        out = self.final_layer(e_four)
        
        return out


# In[23]:


model = UNet(in_channels=3, out_channels=2).cuda()

# # Train

# In[24]:


def loss_funtion(logits, target):
    target = target.squeeze_(dim=1)
    
    return F.cross_entropy(logits, target)


# In[25]:


learn = Learner(data, model, loss_func=loss_funtion, metrics=dice)


# In[26]:


learn.lr_find()


# In[27]:


learn.recorder.plot()


# In[28]:


learn.fit_one_cycle(5, slice(5e-4))


# In[29]:


learn.show_results(rows=2, figsize=(9, 11))