import argparse

import numpy as np

import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import math

import cv2
from my_model_in import *
import random
from lmfit import Model


try:
    from torchvision.transforms.functional import InterpolationMode

    has_interpolation_mode = True
except ImportError:
    has_interpolation_mode = False
from PIL import Image

from my_model_in import *

parser = argparse.ArgumentParser(description="resnet_teacher")
parser.add_argument("--model_name", default="my_resnet18_pad", type=str)
parser.add_argument("--pretrained", default="T", type=str)
parser.add_argument("--interpolation", default="bilinear", type=str)
parser.add_argument("--memostr", default="", type=str)
parser.add_argument("--img_size", default=224, type=int)
parser.add_argument("--crop_pct", default=0.875, type=float)
parser.add_argument("--mode", default="test", type=str)
parser.add_argument("--type", default="resnet", type=str)
parser.add_argument("--stat", default="default", type=str)
parser.add_argument("--hp_bs", type=int, default=256)
parser.add_argument("--hp_nw", type=int, default=4)
parser.add_argument("--hp_id", type=int, default=0)

args = parser.parse_args()

if args.mode in ["grad_dAdi", "grad_dAdi_m0s1", "grad_dAdi_vit"]:
    args.hp_bs = 1


device = torch.device("cuda:" + str(args.hp_id) if torch.cuda.is_available() else "cpu")


if hasattr(Image, "Resampling"):
    _pil_interpolation_to_str = {
        Image.Resampling.NEAREST: "nearest",
        Image.Resampling.BILINEAR: "bilinear",
        Image.Resampling.BICUBIC: "bicubic",
        Image.Resampling.BOX: "box",
        Image.Resampling.HAMMING: "hamming",
        Image.Resampling.LANCZOS: "lanczos",
    }
else:
    _pil_interpolation_to_str = {
        Image.NEAREST: "nearest",
        Image.BILINEAR: "bilinear",
        Image.BICUBIC: "bicubic",
        Image.BOX: "box",
        Image.HAMMING: "hamming",
        Image.LANCZOS: "lanczos",
    }

_str_to_pil_interpolation = {b: a for a, b in _pil_interpolation_to_str.items()}

if has_interpolation_mode:
    _torch_interpolation_to_str = {
        InterpolationMode.NEAREST: "nearest",
        InterpolationMode.BILINEAR: "bilinear",
        InterpolationMode.BICUBIC: "bicubic",
        InterpolationMode.BOX: "box",
        InterpolationMode.HAMMING: "hamming",
        InterpolationMode.LANCZOS: "lanczos",
    }
    _str_to_torch_interpolation = {b: a for a, b in _torch_interpolation_to_str.items()}
else:
    _pil_interpolation_to_torch = {}
    _torch_interpolation_to_str = {}


def str_to_pil_interp(mode_str):
    return _str_to_pil_interpolation[mode_str]


def str_to_interp_mode(mode_str):
    if has_interpolation_mode:
        return _str_to_torch_interpolation[mode_str]
    else:
        return _str_to_pil_interpolation[mode_str]


def interp_mode_to_str(mode):
    if has_interpolation_mode:
        return _torch_interpolation_to_str[mode]
    else:
        return _pil_interpolation_to_str[mode]


scale_size = math.floor(args.img_size / args.crop_pct)

if args.stat in ["default"]:
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    stat_mean = IMAGENET_DEFAULT_MEAN
    stat_std = IMAGENET_DEFAULT_STD
elif args.stat in ["inception"]:
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
    stat_mean = IMAGENET_INCEPTION_MEAN
    stat_std = IMAGENET_INCEPTION_STD
elif args.stat in ["zerounit"]:
    IMAGENET_ZEROUNIT_MEAN = (0.0, 0.0, 0.0)
    IMAGENET_ZEROUNIT_STD = (1.0, 1.0, 1.0)
    stat_mean = IMAGENET_ZEROUNIT_MEAN
    stat_std = IMAGENET_ZEROUNIT_STD
elif args.stat in ["halfhalf"]:
    IMAGENET_ZEROUNIT_MEAN = (0.5, 0.5, 0.5)
    IMAGENET_ZEROUNIT_STD = (0.5, 0.5, 0.5)
    stat_mean = IMAGENET_ZEROUNIT_MEAN
    stat_std = IMAGENET_ZEROUNIT_STD


if args.mode in ["test", "grad_dAdi", "grad_dAdi_vit"]:
    transform = transforms.Compose(
        [
            ### transforms.Resize(256),
            ### transforms.Resize(256, interpolation=str_to_interp_mode(args.interpolation)),
            ### transforms.CenterCrop(224),
            transforms.Resize(
                scale_size, interpolation=str_to_interp_mode(args.interpolation)
            ),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            ### transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=stat_mean, std=stat_std),
        ]
    )

if args.mode in ["test_m0s1", "grad_dAdi_m0s1"]:
    transform = transforms.Compose(
        [
            ### transforms.Resize(256),
            ### transforms.Resize(256, interpolation=str_to_interp_mode(args.interpolation)),
            ### transforms.CenterCrop(224),
            transforms.Resize(
                scale_size, interpolation=str_to_interp_mode(args.interpolation)
            ),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ]
    )

val_set = torchvision.datasets.ImageNet(
    root="~/imagenet/",
    transform=transform,
    split="val",
)
val_loader = data.DataLoader(
    val_set,
    batch_size=args.hp_bs,
    shuffle=False,
    num_workers=args.hp_nw,
    pin_memory=True,
)


if args.mode in ["test", "test_m0s1"]:
    if args.pretrained == "T":
        pretrained = True
    elif args.pretrained == "F":
        pretrained = False
    model = timm.create_model(args.model_name, pretrained=pretrained, num_classes=1000)

    model = model.to(device)
    model.eval()

    correct_top1 = 0.0
    correct_top5 = 0.0
    total = 50000.0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)  # [100, 3, 224, 224]
            labels = labels.to(device)  # [100]
            outputs = model(images)

            # top 1
            _, pred = torch.max(outputs, 1)
            correct_top1 += (pred == labels).sum().item()

            # top 5
            _, rank5 = outputs.topk(5, 1, True, True)
            rank5 = rank5.t()
            correct5 = rank5.eq(labels.view(1, -1).expand_as(rank5))
            for k in range(5):
                correct_k = correct5[k, :].sum()
                correct_top5 += correct_k.item()

    # finally,
    print("top-1 percentage :  {0:0.3f}%".format(correct_top1 / total * 100.0))
    print("top-5 percentage :  {0:0.3f}%".format(correct_top5 / total * 100.0))


if args.mode in ["grad_dAdi", "grad_dAdi_m0s1"]:
    if args.pretrained == "T":
        pretrained = True
    elif args.pretrained == "F":
        pretrained = False
    model = timm.create_model(args.model_name, pretrained=pretrained, num_classes=1000)

    model = model.to(device)
    model.eval()

    # hooker funciton
    def save_gradient(grad):
        return grad

    dAdi_np_cum = np.zeros((args.img_size, args.img_size))

    for idx, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.to(device)  # [100, 3, 224, 224]
        targets = targets.to(device)  # [100]
        outputs = model(inputs)

        feature_dict = {}

        # img도 hook
        x = inputs
        x.requires_grad = True
        x.register_hook(save_gradient)
        feature_dict["img"] = x

        # for ResNet
        for name, module in model._modules.items():
            if name not in ["fc", "classifier"]:
                x = module(x)
                x.register_hook(save_gradient)
                feature_dict[name] = x

        ###
        if args.type in ["eff"]:
            A_ijk = feature_dict["bn2"]
        elif args.type in ["nf"]:
            A_ijk = feature_dict["final_act"]
        elif args.type in ["reg"]:
            A_ijk = feature_dict["final_conv"]
        elif args.type in ["tre"]:
            A_ijk = feature_dict["body"]
        elif args.type in ["den", "rex", "dpn"]:
            A_ijk = feature_dict["features"]
        elif args.type in ["dar"]:
            A_ijk = feature_dict["stages"]
        else:
            # for ResNet
            A_ijk = feature_dict["layer4"]  # 1, 512, 7, 7
        ###

        _, _, H, W = A_ijk.size()  # 7, 7
        c_x = int(H / 2)  # 3
        c_y = int(W / 2)
        A = A_ijk[0, :, c_x, c_y].mean()  # channel mean, center가 3, 3.
        A.backward(retain_graph=True)

        dAdi = inputs.grad  # 1, 3, 224, 224
        dAdi_np = dAdi.cpu().data.numpy()
        dAdi_np = dAdi_np.mean(axis=(0, 1))  # channel mean이니까
        dAdi_np = np.maximum(dAdi_np, 0)  # 224, 224
        dAdi_np_cum += dAdi_np

    # for lmfit
    def gaussian_2d(xy_mesh, amp, xc, yc, sigma_x, sigma_y):
        # unpack 1D list into 2D x and y coords
        (x, y) = xy_mesh

        # make the 2D Gaussian matrix
        gauss = (
            amp
            * np.exp(
                -(
                    (x - xc) ** 2 / (2 * sigma_x ** 2)
                    + (y - yc) ** 2 / (2 * sigma_y ** 2)
                )
            )
            / (2 * np.pi * sigma_x * sigma_y)
        )

        # flatten the 2D Gaussian down to 1D
        return np.ravel(gauss)

    dAdi_np_cum = dAdi_np_cum / np.sum(dAdi_np_cum)

    x = np.arange(args.img_size)
    y = np.arange(args.img_size)
    xy_mesh = np.meshgrid(x, y)
    noise = dAdi_np_cum
    amp = 1
    xc, yc = np.median(x), np.median(y)
    sigma_x, sigma_y = x[-1] / 10, y[-1] / 6
    guess_vals = [amp * 2, xc * 0.8, yc * 0.8, sigma_x / 1.5, sigma_y / 1.5]
    # tell LMFIT what fn you want to fit, then fit, starting iteration with guess values
    lmfit_model = Model(gaussian_2d)
    lmfit_result = lmfit_model.fit(
        np.ravel(noise),
        xy_mesh=xy_mesh,
        amp=guess_vals[0],
        xc=guess_vals[1],
        yc=guess_vals[2],
        sigma_x=guess_vals[3],
        sigma_y=guess_vals[4],
    )
    # again, calculate R-squared
    lmfit_Rsquared = 1 - lmfit_result.residual.var() / np.var(noise)
    print("Fit R-squared:", lmfit_Rsquared, "\n")
    print(lmfit_result.fit_report())

    # save png
    dAdi_np_cum -= np.min(dAdi_np_cum)
    dAdi_np_cum = np.uint8(dAdi_np_cum * 255.0 / np.max(dAdi_np_cum))
    cv2.imwrite(
        "../p_result/"
        + "/erf_dAdi_new_in_"
        + args.model_name
        + "_"
        + args.pretrained
        + ".png",
        dAdi_np_cum,
    )


if args.mode in ["grad_dAdi_vit"]:
    if args.pretrained == "T":
        pretrained = True
    elif args.pretrained == "F":
        pretrained = False
    model = timm.create_model(args.model_name, pretrained=pretrained, num_classes=1000)

    if args.type in ["vreli"]:  # before putting cuda, initialize Mlp.
        for blk in model.blocks:
            blk.attn.rel_pos.mlp.fc1 = nn.Linear(
                in_features=blk.attn.rel_pos.mlp.fc1.in_features,
                out_features=blk.attn.rel_pos.mlp.fc1.out_features,
            )
            blk.attn.rel_pos.mlp.fc2 = nn.Linear(
                in_features=blk.attn.rel_pos.mlp.fc2.in_features,
                out_features=blk.attn.rel_pos.mlp.fc2.out_features,
            )

    model = model.to(device)
    model.eval()

    # hooker funciton
    def save_gradient(grad):
        return grad

    dAdi_np_cum = np.zeros((args.img_size, args.img_size))

    if args.type in ["vrelz"]:
        ### prepare my_zero_rel_pos
        num_heads = model.blocks[0].attn.num_heads  # 6  # medium 8
        (
            patch_size_H,
            patch_size_W,
        ) = model.patch_embed.patch_size  # 16, 16  # medium 16, 16
        num_all_patches = int(
            (args.img_size * args.img_size) / (patch_size_H * patch_size_W)
        )  # 196  # medium 196
        my_zero_rel_pos = torch.zeros(
            (1, num_heads, num_all_patches, num_all_patches)
        ).to(
            device
        )  # torch.Tensor
        ###

        ### remove all rel_pos
        for blk in model.blocks:
            blk.attn.rel_pos = None
        ###
    elif args.type in ["beitz"]:
        ### prepare my_zero_rel_pos
        num_heads = model.blocks[0].attn.num_heads  # 12
        patch_size_H, patch_size_W = model.patch_embed.patch_size  # 16, 16
        num_all_patches = int(
            (args.img_size * args.img_size) / (patch_size_H * patch_size_W)
        )  # 16
        my_zero_rel_pos = torch.zeros(
            (1, num_heads, num_all_patches + 1, num_all_patches + 1)
        ).to(
            device
        )  # torch.Tensor. For beitz, add 1 to have 197.
        ###

        ### remove all rel_pos
        for blk in model.blocks:
            blk.attn.rel_pos = None
        ###
    elif args.type in ["vabsz"]:
        # model.pos_embed  # 1, 50, 384, device cuda, requires_grad True, torch.nn.parameter.Parameter
        model.pos_embed = torch.nn.Parameter(torch.zeros_like(model.pos_embed))
        # After this: 1, 50, 384, device cuda, requires_grad True, torch.nn.parameter.Parameter
    elif args.type in ["swinz"]:
        ### init rel pos
        for lyr in model.layers:
            for blk in lyr.blocks:
                timm.models.layers.trunc_normal_(blk.attn.relative_position_bias_table)
        ###
    elif args.type in ["swin2z"]:
        ### init rel pos
        for lyr in model.layers:
            for blk in lyr.blocks:
                blk.attn.cpb_mlp[0] = torch.nn.Linear(
                    in_features=blk.attn.cpb_mlp[0].in_features,
                    out_features=blk.attn.cpb_mlp[0].out_features,
                    bias=not (blk.attn.cpb_mlp[0].bias is None),
                ).to(device)
                blk.attn.cpb_mlp[2] = torch.nn.Linear(
                    in_features=blk.attn.cpb_mlp[2].in_features,
                    out_features=blk.attn.cpb_mlp[2].out_features,
                    bias=not (blk.attn.cpb_mlp[2].bias is None),
                ).to(device)
        ###

    for idx, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.to(device)  # [100, 3, 224, 224]
        targets = targets.to(device)  # [100]
        outputs = model(inputs)

        feature_dict = {}

        # img도 hook
        if "cait" not in args.model_name:
            x = inputs
            x.requires_grad = True
            x.register_hook(save_gradient)
            feature_dict["img"] = x

        # for ResNet
        for name, module in model._modules.items():
            if args.type in [
                "cait",
                "vrel",
                "vrelc",
                "swinf",
                "vrelz",
                "beitz",
                "vabsz",
                "vreli",
            ]:
                break  # exit this for loop. Instead, use forward_features.

            if (name in ["layers"]) and ("ModuleList" in str(type(module))):  # swin v2
                for sub_module in module:
                    x = sub_module(x)
                continue  # skip below in this loop

            if name not in ["fc", "classifier"]:
                x = module(x)
                x.register_hook(save_gradient)
                feature_dict[name] = x

        ###
        if args.type in ["eff"]:
            A_ijk = feature_dict["bn2"]
        elif args.type in ["nf"]:
            A_ijk = feature_dict["final_act"]
        elif args.type in ["reg"]:
            A_ijk = feature_dict["final_conv"]
        elif args.type in ["tre"]:
            A_ijk = feature_dict["body"]
        elif args.type in ["den", "rex", "dpn"]:
            A_ijk = feature_dict["features"]
        elif args.type in ["dar"]:
            A_ijk = feature_dict["stages"]
        elif args.type in ["vit"]:
            A_ijk = feature_dict["norm"]
        elif args.type in [
            "cait",
            "vrel",
            "vrelc",
            "swinf",
            "vabsz",
            "swinz",
            "swin2z",
        ]:
            inputs.requires_grad = True
            A_ijk = model.forward_features(inputs)
            A_ijk.register_hook(save_gradient)
        elif args.type in ["vrelz"]:
            x = model.patch_embed(inputs)  # 1, 196, 384
            if model.cls_token is not None:  # In fact, this is None.
                x = torch.cat((model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            shared_rel_pos = (
                model.shared_rel_pos.get_bias()
                if model.shared_rel_pos is not None
                else None
            )  # for srel: 1, 6, 196, 196. device cuda. torch.Tensor.  # medium 1, 8, 196, 196

            for blk in model.blocks:
                if (
                    model.grad_checkpointing and not torch.jit.is_scripting()
                ):  # In fact, this is False
                    x = checkpoint(blk, x, shared_rel_pos=my_zero_rel_pos)
                else:
                    x = blk(x, shared_rel_pos=my_zero_rel_pos)
            A_ijk = model.norm(x)  # 1, 196, 384  # checked!
        elif args.type in ["vreli"]:
            x = model.patch_embed(inputs)
            if model.cls_token is not None:  # In fact, this is None.
                x = torch.cat((model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            shared_rel_pos = (
                model.shared_rel_pos.get_bias()
                if model.shared_rel_pos is not None
                else None
            )

            for blk in model.blocks:
                if model.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(blk, x, shared_rel_pos=shared_rel_pos)
                else:
                    x = blk(x, shared_rel_pos=shared_rel_pos)
            A_ijk = model.norm(x)  # checked!
        elif args.type in ["beitz"]:
            x = model.patch_embed(inputs)  # 1, 196, 768
            x = torch.cat(
                (model.cls_token.expand(x.shape[0], -1, -1), x), dim=1
            )  # cls_token: 1, 1, 768  # 1, 197, 768
            if model.pos_embed is not None:  # In fact, this is None.
                x = x + model.pos_embed
            x = model.pos_drop(x)
            rel_pos_bias = (
                model.rel_pos_bias() if model.rel_pos_bias is not None else None
            )  # In fact, this is None.

            for blk in model.blocks:
                if model.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(blk, x, shared_rel_pos_bias=my_zero_rel_pos)
                else:
                    x = blk(x, shared_rel_pos_bias=my_zero_rel_pos)
            A_ijk = model.norm(x)  # 1, 197, 768, checked!
        else:
            # for ResNet
            A_ijk = feature_dict["layer4"]  # 1, 512, 7, 7
        ###

        # For vit,
        _, HW, _ = A_ijk.size()  # [1, 49 768] -> 49
        if args.type in [
            "cait",
            "vrelc",
            "beitz",
            "vabsz",
        ]:  # subtract 1. this is correct. not for (vrel, swinf, swinz, swin2z).
            HW = HW - 1
        # center = int(HW / 2)  # 24
        center = int(HW / 2) - ((HW + 1) % 2) * (int(math.sqrt(HW) / 2) + 1)
        if args.type in [
            "cait",
            "vrelc",
            "beitz",
            "vabsz",
        ]:  # add 1. not for (vrel, swinf, swinz, swin2z).
            center = center + 1
        A = A_ijk[:, center, :].mean()
        A.backward(retain_graph=True)

        dAdi = inputs.grad  # 1, 3, 224, 224
        dAdi_np = dAdi.cpu().data.numpy()
        dAdi_np = dAdi_np.mean(axis=(0, 1))  # channel mean이니까
        dAdi_np = np.maximum(dAdi_np, 0)  # 224, 224
        dAdi_np_cum += dAdi_np

    # for lmfit
    def gaussian_2d(xy_mesh, amp, xc, yc, sigma_x, sigma_y):
        # unpack 1D list into 2D x and y coords
        (x, y) = xy_mesh

        # make the 2D Gaussian matrix
        gauss = (
            amp
            * np.exp(
                -(
                    (x - xc) ** 2 / (2 * sigma_x ** 2)
                    + (y - yc) ** 2 / (2 * sigma_y ** 2)
                )
            )
            / (2 * np.pi * sigma_x * sigma_y)
        )

        # flatten the 2D Gaussian down to 1D
        return np.ravel(gauss)

    dAdi_np_cum = dAdi_np_cum / np.sum(dAdi_np_cum)

    x = np.arange(args.img_size)
    y = np.arange(args.img_size)
    xy_mesh = np.meshgrid(x, y)
    noise = dAdi_np_cum
    amp = 1
    xc, yc = np.median(x), np.median(y)
    sigma_x, sigma_y = x[-1] / 10, y[-1] / 6
    guess_vals = [amp * 2, xc * 0.8, yc * 0.8, sigma_x / 1.5, sigma_y / 1.5]
    # tell LMFIT what fn you want to fit, then fit, starting iteration with guess values
    lmfit_model = Model(gaussian_2d)
    lmfit_result = lmfit_model.fit(
        np.ravel(noise),
        xy_mesh=xy_mesh,
        amp=guess_vals[0],
        xc=guess_vals[1],
        yc=guess_vals[2],
        sigma_x=guess_vals[3],
        sigma_y=guess_vals[4],
    )
    # again, calculate R-squared
    lmfit_Rsquared = 1 - lmfit_result.residual.var() / np.var(noise)
    print("Fit R-squared:", lmfit_Rsquared, "\n")
    print(lmfit_result.fit_report())

    # save png
    dAdi_np_cum -= np.min(dAdi_np_cum)
    dAdi_np_cum = np.uint8(dAdi_np_cum * 255.0 / np.max(dAdi_np_cum))
    cv2.imwrite(
        "../p_result/"
        + "/erf_dAdi_new_in_"
        + args.model_name
        + args.memostr
        + "_"
        + args.pretrained
        + ".png",
        dAdi_np_cum,
    )
