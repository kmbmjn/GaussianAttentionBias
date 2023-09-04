import timm
import torch
import numpy as np
import cv2
import pdb
from lmfit import Model
import math

model_name = "vit_relpos_base_patch16_224"
# model_name = "vit_relpos_medium_patch16_224"
# model_name = "vit_relpos_small_patch16_224"

# model_name = "beit_base_patch16_224"
# model_name = "beit_large_patch16_224"
# model_name = "beit_base_patch16_384"
# model_name = "beit_large_patch16_384"
# model_name = "beit_large_patch16_512"

# model_name = "swin_small_patch4_window7_224"
# model_name = "swinv2_small_window16_256"

# model_name = "vit_srelpos_small_patch16_224"
# model_name = "vit_srelpos_medium_patch16_224"

# model_name = "vit_small_patch16_224"
# model_name = "vit_base_patch16_224"
# model_name = "vit_large_patch16_224"
# model_name = "vit_small_patch16_384"
# model_name = "vit_base_patch16_384"
# model_name = "vit_large_patch16_384"

# model_name = "deit_small_patch16_224"
# model_name = "deit_base_patch16_224"
# model_name = "deit_base_patch16_384"


pretrained = True
# pretrained = False

model = timm.create_model(model_name, pretrained=pretrained)


# abs pos, global
# if True:
if False:
    bias_0 = model.pos_embed  # 1, 197, 384
    _, len_bias, _ = bias_0.size()  # 197
    len_patch = int(math.sqrt(len_bias - 1))  # 14

    bias_0_select = bias_0[:, 1:, :].mean(dim=(0, 2)).reshape((len_patch, len_patch))

    # save png
    result_np = bias_0_select.detach().numpy()
    result_np -= np.min(result_np)
    if np.max(result_np) != 0.0:
        result_np = np.uint8(result_np * 255.0 / np.max(result_np))
        cv2.imwrite(
            "abs_pos_result/global_result_np_"
            + model_name
            + "_"
            + str(pretrained)
            + ".png",
            result_np,
        )

# abs pos, local
# if True:
if False:
    bias_0 = model.pos_embed  # 1, 197, 384
    _, len_bias, len_channel = bias_0.size()  # 197 & 384
    len_patch = int(math.sqrt(len_bias - 1))  # 14

    for cth in range(len_channel):
        bias_0_select = bias_0[0, 1:, cth].reshape(
            (len_patch, len_patch)
        )  # [196] -> [14, 14]

        # save png
        result_np = bias_0_select.detach().numpy()
        result_np -= np.min(result_np)
        if np.max(result_np) != 0.0:
            result_np = np.uint8(result_np * 255.0 / np.max(result_np))
            cv2.imwrite(
                "abs_pos_result/local_result_np_"
                + model_name
                + "_"
                + str(pretrained)
                + "_cth"
                + str(cth)
                + ".png",
                result_np,
            )


# rel pos
# save by patch
# if True:
if False:
    lth = 0
    if ("vit" in model_name) and ("srelpos" in model_name):
        bias_0 = model.shared_rel_pos.get_bias()  # 1, 6, 196, 196
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias))
    elif ("vit" in model_name) and ("cls" in model_name):
        bias_0 = model.blocks[lth].attn.rel_pos.get_bias()  # 1, 8, 197, 197
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias - 1))
    elif "vit" in model_name:
        bias_0 = model.blocks[lth].attn.rel_pos.get_bias()  # 1, 6, 196, 196
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias))
    elif "beit" in model_name:
        bias_0 = model.blocks[lth].attn._get_rel_pos_bias()  # 1, 24, 197, 197
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias - 1))
    elif "swin" in model_name:
        bias_0 = (
            model.layers[0].blocks[0].attn._get_rel_pos_bias()
        )  # 1, 3, 49, 49, but 3 can become 24 in other layer.
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias))

    for index in range(len_bias):
        if ("vit" in model_name) and ("srelpos" in model_name):
            bias_0_select = (
                bias_0[:, :, index, :].mean(dim=(0, 1)).reshape((len_patch, len_patch))
            )  # 196 -> 14, 14
        elif ("vit" in model_name) and ("cls" in model_name):
            bias_0_select = (
                bias_0[:, :, index, 1:].mean(dim=(0, 1)).reshape((len_patch, len_patch))
            )
        elif "vit" in model_name:
            bias_0_select = (
                bias_0[:, :, index, :].mean(dim=(0, 1)).reshape((len_patch, len_patch))
            )  # 196 -> 14, 14
        elif "beit" in model_name:
            bias_0_select = (
                bias_0[:, :, index, 1:].mean(dim=(0, 1)).reshape((len_patch, len_patch))
            )
        elif "swin" in model_name:
            bias_0_select = (
                bias_0[:, :, index, :].mean(dim=(0, 1)).reshape((len_patch, len_patch))
            )

        # save png
        result_np = bias_0_select.detach().numpy()
        result_np -= np.min(result_np)
        if np.max(result_np) != 0.0:
            result_np = np.uint8(result_np * 255.0 / np.max(result_np))
            cv2.imwrite(
                "rel_pos_result/result_np_"
                + model_name
                + "_"
                + str(pretrained)
                + "_lth"
                + str(lth)
                + "_index"
                + str(index)
                + ".png",
                result_np,
            )


# rel pos
# save by patch
# gaussian
# if True:
if False:
    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")

    num_heads = model.blocks[0].attn.num_heads  # 6
    patch_size_H, patch_size_W = model.patch_embed.patch_size  # 16, 16
    img_size = int(model_name[-3:])  # 224
    num_all_patches = int((img_size * img_size) / (patch_size_H * patch_size_W))  # 196
    len_patch = int(math.sqrt(num_all_patches))  # 14

    relative_coords_H = torch.arange(
        -(len_patch - 1), len_patch, dtype=torch.float32
    )  # value -13. ~ 0. ~ 13. re_gr F
    relative_coords_W = torch.arange(-(len_patch - 1), len_patch, dtype=torch.float32)
    relative_coords_table = torch.stack(
        torch.meshgrid([relative_coords_H, relative_coords_W])
    ).to(
        device
    )  # size 2, 27, 27. re_gr F
    sigma_param = torch.nn.Parameter(torch.Tensor([5.0])).to(
        device
    )  # by default, it becomes requires_grad True and updated by GD.
    amplitude_param = torch.nn.Parameter(torch.Tensor([1e0])).to(device)
    Gaussian_wide = (
        amplitude_param
        * amplitude_param
        * torch.exp(
            -1.0
            * (
                relative_coords_table[0, :, :] ** 2.0
                + relative_coords_table[1, :, :] ** 2.0
            )
            / (2.0 * (sigma_param ** 2.0))
        )
    )  # 27, 27
    Gaussian_pos = torch.stack(
        [
            Gaussian_wide[
                start_H : start_H + len_patch, start_W : start_W + len_patch
            ].reshape(-1)
            for start_H in range(0, len_patch)
            for start_W in range(0, len_patch)
        ]
    ).expand(
        1, num_heads, num_all_patches, num_all_patches
    )  # 196, 196 -> 1, 6, 196, 196, cuda

    lth = 0
    if ("vit" in model_name) and ("srelpos" in model_name):
        bias_0 = model.shared_rel_pos.get_bias()  # 1, 6, 196, 196
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias))
    elif ("vit" in model_name) and ("cls" in model_name):
        bias_0 = model.blocks[lth].attn.rel_pos.get_bias()  # 1, 8, 197, 197
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias - 1))
    elif "vit" in model_name:
        bias_0 = model.blocks[lth].attn.rel_pos.get_bias()  # 1, 6, 196, 196
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias))
    elif "beit" in model_name:
        bias_0 = model.blocks[lth].attn._get_rel_pos_bias()  # 1, 24, 197, 197
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias - 1))
    elif "swin" in model_name:
        bias_0 = (
            model.layers[0].blocks[0].attn._get_rel_pos_bias()
        )  # 1, 3, 49, 49, but 3 can become 24 in other layer.
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias))

    for index in range(len_bias):
        if ("vit" in model_name) and ("srelpos" in model_name):
            bias_0_select = (
                bias_0[:, :, index, :].mean(dim=(0, 1)).reshape((len_patch, len_patch))
            )  # 196 -> 14, 14
        elif ("vit" in model_name) and ("cls" in model_name):
            bias_0_select = (
                bias_0[:, :, index, 1:].mean(dim=(0, 1)).reshape((len_patch, len_patch))
            )
        elif "vit" in model_name:
            bias_0_select = (
                bias_0[:, :, index, :].mean(dim=(0, 1)).reshape((len_patch, len_patch))
            )  # 196 -> 14, 14
        elif "beit" in model_name:
            bias_0_select = (
                bias_0[:, :, index, 1:].mean(dim=(0, 1)).reshape((len_patch, len_patch))
            )
        elif "swin" in model_name:
            bias_0_select = (
                bias_0[:, :, index, :].mean(dim=(0, 1)).reshape((len_patch, len_patch))
            )

        # save png
        result_np = bias_0_select.detach().numpy()
        result_np -= np.min(result_np)
        if np.max(result_np) != 0.0:
            result_np = np.uint8(result_np * 255.0 / np.max(result_np))
            cv2.imwrite(
                "rel_pos_result/result_np_"
                + model_name
                + "_"
                + str(pretrained)
                + "_lth"
                + str(lth)
                + "_index"
                + str(index)
                + ".png",
                result_np,
            )


# rel pos
# save by patch and head
# if True:
if False:
    lth = 0
    if ("vit" in model_name) and ("srelpos" in model_name):
        bias_0 = model.shared_rel_pos.get_bias()  # 1, 6, 196, 196
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias))
    elif ("vit" in model_name) and ("cls" in model_name):
        bias_0 = model.blocks[lth].attn.rel_pos.get_bias()  # 1, 8, 197, 197
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias - 1))
    elif "vit" in model_name:
        bias_0 = model.blocks[lth].attn.rel_pos.get_bias()  # 1, 6, 196, 196
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias))
    elif "beit" in model_name:
        bias_0 = model.blocks[lth].attn._get_rel_pos_bias()  # 1, 24, 197, 197
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias - 1))
    elif "swin" in model_name:
        bias_0 = (
            model.layers[0].blocks[0].attn._get_rel_pos_bias()
        )  # 1, 3, 49, 49, but 3 can become 24 in other layer.
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias))

    for index in range(len_bias):
        if ("vit" in model_name) and ("srelpos" in model_name):
            bias_0_select = (
                bias_0[:, :, index, :].mean(dim=(0, 1)).reshape((len_patch, len_patch))
            )  # 196 -> 14, 14
        elif ("vit" in model_name) and ("cls" in model_name):
            bias_0_select = (
                bias_0[:, :, index, 1:].mean(dim=(0, 1)).reshape((len_patch, len_patch))
            )
        elif "vit" in model_name:
            bias_0_select = bias_0[0, :, index, :].reshape(
                (-1, len_patch, len_patch)
            )  # 6, 14, 14
        elif "beit" in model_name:
            bias_0_select = (
                bias_0[:, :, index, 1:].mean(dim=(0, 1)).reshape((len_patch, len_patch))
            )
        elif "swin" in model_name:
            bias_0_select = (
                bias_0[:, :, index, :].mean(dim=(0, 1)).reshape((len_patch, len_patch))
            )

        num_head, _, _ = bias_0_select.size()  # 6
        for hth in range(num_head):
            # save png
            result_np = bias_0_select[hth, :, :].detach().numpy()
            result_np -= np.min(result_np)
            if np.max(result_np) != 0.0:
                result_np = np.uint8(result_np * 255.0 / np.max(result_np))
                cv2.imwrite(
                    "rel_pos_result_head/result_np_"
                    + model_name
                    + "_"
                    + str(pretrained)
                    + "_lth"
                    + str(lth)
                    + "_index"
                    + str(index)
                    + "_head"
                    + str(hth)
                    + ".png",
                    result_np,
                )


# rel pos
# save by patch and head
# head diff
# if True:
if False:
    lth = 0
    if ("vit" in model_name) and ("srelpos" in model_name):
        bias_0 = model.shared_rel_pos.get_bias()  # 1, 6, 196, 196
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias))
    elif ("vit" in model_name) and ("cls" in model_name):
        bias_0 = model.blocks[lth].attn.rel_pos.get_bias()  # 1, 8, 197, 197
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias - 1))
    elif "vit" in model_name:
        bias_0 = model.blocks[lth].attn.rel_pos.get_bias()  # 1, 6, 196, 196
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias))
    elif "beit" in model_name:
        bias_0 = model.blocks[lth].attn._get_rel_pos_bias()  # 1, 24, 197, 197
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias - 1))
    elif "swin" in model_name:
        bias_0 = (
            model.layers[0].blocks[0].attn._get_rel_pos_bias()
        )  # 1, 3, 49, 49, but 3 can become 24 in other layer.
        _, _, len_bias, _ = bias_0.size()
        len_patch = int(math.sqrt(len_bias))

    for index in range(len_bias):
        if ("vit" in model_name) and ("srelpos" in model_name):
            bias_0_select = (
                bias_0[:, :, index, :].mean(dim=(0, 1)).reshape((len_patch, len_patch))
            )  # 196 -> 14, 14
        elif ("vit" in model_name) and ("cls" in model_name):
            bias_0_select = (
                bias_0[:, :, index, 1:].mean(dim=(0, 1)).reshape((len_patch, len_patch))
            )
        elif "vit" in model_name:
            bias_0_select = bias_0[0, :, index, :].reshape(
                (-1, len_patch, len_patch)
            )  # 6, 14, 14
        elif "beit" in model_name:
            bias_0_select = (
                bias_0[:, :, index, 1:].mean(dim=(0, 1)).reshape((len_patch, len_patch))
            )
        elif "swin" in model_name:
            bias_0_select = (
                bias_0[:, :, index, :].mean(dim=(0, 1)).reshape((len_patch, len_patch))
            )

        bias_0_select_mean = (
            bias_0[:, :, index, :].mean(dim=(0, 1)).reshape((len_patch, len_patch))
        )  # 196 -> 14, 14

        num_head, _, _ = bias_0_select.size()  # 6
        for hth in range(num_head):
            # save png
            head_diff = bias_0_select_mean - bias_0_select[hth, :, :]

            result_np = head_diff.detach().numpy()  # 14, 14
            result_np -= np.min(result_np)
            if np.max(result_np) != 0.0:
                result_np = np.uint8(result_np * 255.0 / np.max(result_np))
                cv2.imwrite(
                    "rel_pos_result_head_diff/result_np_"
                    + model_name
                    + "_"
                    + str(pretrained)
                    + "_lth"
                    + str(lth)
                    + "_index"
                    + str(index)
                    + "_head"
                    + str(hth)
                    + ".png",
                    result_np,
                )

# rel pos
# save by layer
# if True:
if False:
    index_center = 90
    for lth in range(len(model.blocks)):
        bias_0 = model.blocks[lth].attn.rel_pos.get_bias()  # 1, 6, 196, 196
        bias_0_select = (
            bias_0[:, :, index_center, :].mean(dim=(0, 1)).reshape((14, 14))
        )  # 196 -> 14, 14

        # save png
        result_np = bias_0_select.detach().numpy()
        result_np -= np.min(result_np)
        result_np = np.uint8(result_np * 255.0 / np.max(result_np))
        cv2.imwrite("result_np_" + model_name + "_layer" + str(lth) + ".png", result_np)

# for lmfit
# if True:
if False:

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

    if "224" in model_name:
        index_center = 90
    elif "384" in model_name:
        index_center = 276
    elif "512" in model_name:
        index_center = 496

    rsq_list = []

    for lth in range(len(model.blocks)):
        if ("vit" in model_name) and ("srelpos" in model_name):
            bias_0 = model.shared_rel_pos.get_bias()  # 1, 6, 196, 196
            _, _, len_bias, _ = bias_0.size()
            len_patch = int(math.sqrt(len_bias))
        elif ("vit" in model_name) and ("cls" in model_name):
            bias_0 = model.blocks[lth].attn.rel_pos.get_bias()  # 1, 8, 197, 197
            _, _, len_bias, _ = bias_0.size()
            len_patch = int(math.sqrt(len_bias - 1))
        elif "vit" in model_name:
            bias_0 = model.blocks[lth].attn.rel_pos.get_bias()  # 1, 6, 196, 196
            _, _, len_bias, _ = bias_0.size()
            len_patch = int(math.sqrt(len_bias))
        elif "beit" in model_name:
            bias_0 = model.blocks[lth].attn._get_rel_pos_bias()  # 1, 24, 197, 197
            _, _, len_bias, _ = bias_0.size()
            len_patch = int(math.sqrt(len_bias - 1))

        if ("vit" in model_name) and ("srelpos" in model_name):
            bias_0_select = (
                bias_0[:, :, index_center, :]
                .mean(dim=(0, 1))
                .reshape((len_patch, len_patch))
            )  # 196 -> 14, 14
        elif ("vit" in model_name) and ("cls" in model_name):
            bias_0_select = (
                bias_0[:, :, index_center, 1:]
                .mean(dim=(0, 1))
                .reshape((len_patch, len_patch))
            )
        elif "vit" in model_name:
            bias_0_select = (
                bias_0[:, :, index_center, :]
                .mean(dim=(0, 1))
                .reshape((len_patch, len_patch))
            )  # 196 -> 14, 14
        elif "beit" in model_name:
            bias_0_select = (
                bias_0[:, :, index_center, 1:]
                .mean(dim=(0, 1))
                .reshape((len_patch, len_patch))
            )

        result_np = bias_0_select.detach().numpy()

        ### fit to gaussian
        result_np -= np.min(result_np)  # very important.
        # result_np = result_np / np.sum(result_np)  # in fact, this does not affect the result.

        x = np.arange(len_patch)
        y = np.arange(len_patch)
        xy_mesh = np.meshgrid(x, y)
        noise = result_np
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
        print("Fit R-squared:", lmfit_Rsquared)
        print(lmfit_result.fit_report())
        print("")
        print("")
        print("")
        # print(sum(rsq_list) / len(rsq_list))

        if ("vit" in model_name) and ("srelpos" in model_name):
            break  # exit this for loop
