import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import transforms
from utils.senet50_fm.senet50_ft_dims_2048 import senet50_ft
from blocks import *


def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to('cuda')
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
    return torch.mean((dydx_l2norm - 1) ** 2)

class ResidualBlock(nn.Module):
    """Residual Block with optional normalization."""
    def __init__(self, dim_in, dim_out, norm_layer):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(dim_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(dim_out))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=6, norm_layer='IN'):
        super(Generator, self).__init__()

        if norm_layer == 'BN':
            self.norm_layer = lambda num_features: nn.BatchNorm2d(num_features, affine=True, track_running_stats=True)
        elif norm_layer == 'IN':
            self.norm_layer = lambda num_features: nn.InstanceNorm2d(num_features, affine=True, track_running_stats=False)

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(self.norm_layer(conv_dim))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(self.norm_layer(curr_dim*2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, norm_layer=self.norm_layer))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(self.norm_layer(curr_dim//2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        # layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.

        return self.main(x)

def generate(path='', layers=64, norm_layer='IN'):
    model = Generator(layers, norm_layer=norm_layer)
    if path:
        state_dic = torch.load(path, map_location=torch.device('cuda'))
        model.load_state_dict(state_dic['state_dict'])
        # model.load_state_dict(state_dic)
    return model


# class Discriminator_STGAN(nn.Module):
#     def __init__(self, image_size=224, conv_dim=64, fc_dim=1024, n_layers=6):
#         super(Discriminator_STGAN, self).__init__()
#         layers = []
#         in_channels = 3
#         for i in range(n_layers):
#             layers.append(nn.Sequential(
#                 nn.Conv2d(in_channels, conv_dim * 2 ** i, 4, 2, 1),
#                 nn.InstanceNorm2d(conv_dim * 2 ** i, affine=True, track_running_stats=True),
#                 nn.LeakyReLU(negative_slope=0.2, inplace=True)
#             ))
#             in_channels = conv_dim * 2 ** i
#         self.conv = nn.Sequential(*layers)
#         feature_size = image_size // 2**n_layers
#         self.fc_adv = nn.Sequential(
#             nn.Linear(conv_dim * 2 ** (n_layers - 1) * feature_size ** 2, fc_dim),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Linear(fc_dim, 1)
#         )
#
#     def forward(self, x):
#         y = self.conv(x)
#         y = y.view(y.size()[0], -1)
#         logit_adv = self.fc_adv(y)
#         return logit_adv
#
# def discriminator_stgan(path='', layers=64, repeat_num=5):
#     model = Discriminator_STGAN(conv_dim=layers, n_layers=repeat_num)
#     if path:
#         state_dic = torch.load(path, map_location=torch.device('cuda'))
#         model.load_state_dict(state_dic['state_dict'])
#     return model


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, conv_dim=64,  repeat_num=6, norm_layer='IN'):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            if norm_layer == 'IN':
                layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=False))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=0, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv(h)
        return out_src.view(out_src.size(0), out_src.size(1))
        # return out_src

def discriminator(path='', layers=64, repeat_num=6, norm_layer='IN'):
    model = Discriminator(layers, repeat_num=repeat_num, norm_layer=norm_layer)
    if path:
        state_dic = torch.load(path, map_location=torch.device('cuda'))
        model.load_state_dict(state_dic['state_dict'])
        # model.load_state_dict(state_dic)
    return model


class NLayerDiscriminator(nn.Module):
    def __init__(self,
            input_ch = 3,
            base_ch = 64,
            max_ch = 1024,
            depth = 4,
            norm_type = 'none',
            relu_type = 'LeakyReLU',
            ):
        super().__init__()

        nargs = {'norm_type': norm_type, 'relu_type': relu_type}
        self.norm_type = norm_type
        self.input_ch = input_ch

        self.model = []
        self.model.append(ConvLayer(input_ch, base_ch, norm_type='none', relu_type=relu_type))
        for i in range(depth):
            cin  = min(base_ch * 2**(i), max_ch)
            cout = min(base_ch * 2**(i+1), max_ch)
            self.model.append(ConvLayer(cin, cout, scale='down_avg', **nargs))
        self.model = nn.Sequential(*self.model)
        self.score_out = ConvLayer(cout, 1, use_pad=False)

    def forward(self, x, return_feat=False):
        ret_feats = []
        for idx, m in enumerate(self.model):
            x = m(x)
            ret_feats.append(x)
        x = self.score_out(x)
        if return_feat:
            return x, ret_feats
        else:
            return x


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_ch, base_ch=64, n_layers=3, norm_type='none', relu_type='LeakyReLU', num_D=4):
        super().__init__()

        self.D_pool = nn.ModuleList()
        for i in range(num_D):
            netD = NLayerDiscriminator(input_ch, base_ch, depth=n_layers, norm_type=norm_type, relu_type=relu_type)
            self.D_pool.append(netD)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input, return_feat=False):
        results = []
        for netd in self.D_pool:
            output = netd(input, return_feat)
            results.append(output)
            # Downsample input
            input = self.downsample(input)
        return results



class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

     """

    def __init__(self, feature_model_extractor_node: str) -> None:
        super(ContentLoss, self).__init__()
        # Get the name of the specified feature extraction node
        self.feature_model_extractor_node = feature_model_extractor_node
        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(pretrained=True)
        # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
        # set to validation mode
        self.feature_extractor.eval()

        # The preprocessing method of the input data. This is the VGG model preprocessing method of the ImageNet dataset.
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, xr_tensor: torch.Tensor, xf_tensor: torch.Tensor) -> torch.Tensor:
        # Standardized operations

        xr_tensor = self.normalize(xr_tensor/2+0.5)
        xf_tensor = self.normalize(xf_tensor/2+0.5)


        xr_feature = self.feature_extractor(xr_tensor)[self.feature_model_extractor_node]
        xf_feature = self.feature_extractor(xf_tensor)[self.feature_model_extractor_node]

        # Find the feature map difference between the two images
        content_loss = F.mse_loss(xr_feature, xf_feature)

        return content_loss


class FaceMatchingLoss(nn.Module):

    def __init__(self) -> None:
        super(FaceMatchingLoss, self).__init__()
        self.matcher = senet50_ft(weights_path='./weights/matcher_for_training/senet50_fm_ft_dims_2048.pth')
        self.matcher.eval()
        self.normalize = transforms.Normalize([131.0912, 103.8827, 91.4953], [1, 1, 1])
        for parameter in self.matcher.parameters():
            parameter.requires_grad = False

    def forward(self, xr_tensor: torch.Tensor, xf_tensor: torch.Tensor) -> torch.Tensor:
        xr_tensor = self.normalize((xr_tensor/2+0.5) * 255)
        xf_tensor = self.normalize((xf_tensor/2+0.5) * 255)

        xr_feature = self.matcher(xr_tensor)
        xf_feature = self.matcher(xf_tensor)

        matching_loss = F.mse_loss(xr_feature, xf_feature)

        return matching_loss


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss