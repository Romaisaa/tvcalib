
from kornia.utils import create_meshgrid
from kornia.geometry.linalg import transform_points
import torch.nn.functional as F
import torch

Tensor = torch.Tensor
tensor = torch.tensor


def homography_warp(
    patch_src,
    src_homo_dst,
    dsize,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=False,
    normalized_coordinates=True,
    normalized_homography=True,
):
    r"""Warp image patches or tensors by normalized 2D homographies.

    See :class:`~kornia.geometry.warp.HomographyWarper` for details.

    Args:
        patch_src: The image or tensor to warp. Should be from source of shape :math:`(N, C, H, W)`.
        src_homo_dst: The homography or stack of homographies from destination to source of shape :math:`(N, 3, 3)`.
        dsize:
          if homography normalized: The height and width of the image to warp.
          if homography not normalized: size of the output image (height, width).
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.
        normalized_coordinates: Whether the homography assumes [-1, 1] normalized coordinates or not.
        normalized_homography: show is homography normalized.

    Return:
        Patch sampled at locations from source to destination.

    Example:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> homography = torch.eye(3).view(1, 3, 3)
        >>> output = homography_warp(input, homography, (32, 32))

    Example
        >>> img = torch.rand(1, 4, 5, 6)
        >>> H = torch.eye(3)[None]
        >>> out = homography_warp(img, H, (4, 2), align_corners=True, normalized_homography=False)
        >>> print(out.shape)
        torch.Size([1, 4, 4, 2])
    """
    if not src_homo_dst.device == patch_src.device:
        raise TypeError(
            f"Patch and homography must be on the same device. Got patch.device: {patch_src.device} "
            f"src_H_dst.device: {src_homo_dst.device}."
        )
    if normalized_homography:
        height, width = dsize
        grid = create_meshgrid(
            height, width, normalized_coordinates=normalized_coordinates, device=patch_src.device, dtype=patch_src.dtype
        )
        warped_grid = warp_grid(grid, src_homo_dst)

        return F.grid_sample(patch_src, warped_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return warp_perspective(
        patch_src, src_homo_dst, dsize, mode="bilinear", padding_mode=padding_mode, align_corners=True
    )


def warp_perspective(
    src,
    M,
    dsize,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=True,
    fill_value=torch.zeros(3),  # needed for jit
):
    r"""Apply a perspective transformation to an image.

    The function warp_perspective transforms the source image using
    the specified matrix:

    .. math::
        \text{dst} (x, y) = \text{src} \left(
        \frac{M^{-1}_{11} x + M^{-1}_{12} y + M^{-1}_{13}}{M^{-1}_{31} x + M^{-1}_{32} y + M^{-1}_{33}} ,
        \frac{M^{-1}_{21} x + M^{-1}_{22} y + M^{-1}_{23}}{M^{-1}_{31} x + M^{-1}_{32} y + M^{-1}_{33}}
        \right )

    Args:
        src: input image with shape :math:`(B, C, H, W)`.
        M: transformation matrix with shape :math:`(B, 3, 3)`.
        dsize: size of the output image (height, width).
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'`` | ``'fill'``.
        align_corners: interpolation flag.
        fill_value: tensor of shape :math:`(3)` that fills the padding area. Only supported for RGB.

    Returns:
        the warped input image :math:`(B, C, H, W)`.

    Example:
       >>> img = torch.rand(1, 4, 5, 6)
       >>> H = torch.eye(3)[None]
       >>> out = warp_perspective(img, H, (4, 2), align_corners=True)
       >>> print(out.shape)
       torch.Size([1, 4, 4, 2])

    .. note::
        This function is often used in conjunction with :func:`get_perspective_transform`.

    .. note::
        See a working example `here <https://kornia.github.io/tutorials/nbs/warp_perspective.html>`_.
    """
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"Input src type is not a Tensor. Got {type(src)}")

    if not isinstance(M, torch.Tensor):
        raise TypeError(f"Input M type is not a Tensor. Got {type(M)}")

    if not len(src.shape) == 4:
        raise ValueError(f"Input src must be a BxCxHxW tensor. Got {src.shape}")

    if not (len(M.shape) == 3 and M.shape[-2:] == (3, 3)):
        raise ValueError(f"Input M must be a Bx3x3 tensor. Got {M.shape}")

    # fill padding is only supported for 3 channels because we can't set fill_value default
    # to None as this gives jit issues.
    if padding_mode == "fill" and fill_value.shape != torch.Size([3]):
        raise ValueError(f"Padding_tensor only supported for 3 channels. Got {fill_value.shape}")

    B, _, H, W = src.size()
    h_out, w_out = dsize

    # we normalize the 3x3 transformation matrix and convert to 3x4
    dst_norm_trans_src_norm: Tensor = normalize_homography(M, (H, W), (h_out, w_out))  # Bx3x3

    src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)  # Bx3x3

    # this piece of code substitutes F.affine_grid since it does not support 3x3
    grid = (
        create_meshgrid(h_out, w_out, normalized_coordinates=True, device=src.device)
        .to(src.dtype)
        .expand(B, h_out, w_out, 2)
    )
    grid = transform_points(src_norm_trans_dst_norm[:, None, None], grid)

    if padding_mode == "fill":
        return _fill_and_warp(src, grid, align_corners=align_corners, mode=mode, fill_value=fill_value)
    return F.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)


def normalize_homography(
    dst_pix_trans_src_pix, dsize_src, dsize_dst,
):
    r"""Normalize a given homography in pixels to [-1, 1].

    Args:
        dst_pix_trans_src_pix: homography/ies from source to destination to be
          normalized. :math:`(B, 3, 3)`
        dsize_src: size of the source image (height, width).
        dsize_dst: size of the destination image (height, width).

    Returns:
        the normalized homography of shape :math:`(B, 3, 3)`.
    """
    if not isinstance(dst_pix_trans_src_pix, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(dst_pix_trans_src_pix)}")

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError(f"Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {dst_pix_trans_src_pix.shape}")

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: Tensor = normal_transform_pixel(src_h, src_w).to(dst_pix_trans_src_pix)

    src_pix_trans_src_norm = _torch_inverse_cast(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: Tensor = normal_transform_pixel(dst_h, dst_w).to(dst_pix_trans_src_pix)

    # compute chain transformations
    dst_norm_trans_src_norm: Tensor = dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm


def normal_transform_pixel(
    height,
    width,
    eps=1e-14,
    device=None,
    dtype=None,
):
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].

    Args:
        height image height.
        width: image width.
        eps: epsilon to prevent divide-by-zero errors

    Returns:
        normalized transform with shape :math:`(1, 3, 3)`.
    """
    tr_mat = tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)  # 3x3

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return tr_mat.unsqueeze(0)  # 1x3x3


def _torch_inverse_cast(input: Tensor) -> Tensor:
    """Helper function to make torch.inverse work with other than fp32/64.

    The function torch.inverse is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply torch.inverse, and cast back to the input dtype.
    """
    if not isinstance(input, Tensor):
        raise AssertionError(f"Input must be Tensor. Got: {type(input)}.")
    dtype: torch.dtype = input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    return torch.inverse(input.to(dtype)).to(input.dtype)