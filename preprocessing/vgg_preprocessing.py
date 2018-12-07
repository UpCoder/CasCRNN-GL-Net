from models.research.slim.preprocessing.vgg_preprocessing import preprocess_image, \
    _aspect_preserving_resize, _random_crop, _mean_image_subtraction, _RESIZE_SIDE_MAX, _RESIZE_SIDE_MIN, \
    _R_MEAN, _G_MEAN, _B_MEAN
import tensorflow as tf


def preprocessing_image_train(nc_roi, art_roi, pv_roi, nc_patch, art_patch, pv_patch, output_height, output_width, resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX):
    resize_side = tf.random_uniform(
        [], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32)

    nc_roi = _aspect_preserving_resize(nc_roi, resize_side)
    art_roi = _aspect_preserving_resize(art_roi, resize_side)
    pv_roi = _aspect_preserving_resize(pv_roi, resize_side)
    nc_patch = _aspect_preserving_resize(nc_patch, resize_side)
    art_patch = _aspect_preserving_resize(art_patch, resize_side)
    pv_patch = _aspect_preserving_resize(pv_patch, resize_side)

    image = _random_crop([nc_roi, art_roi, pv_roi, nc_patch, art_patch, pv_patch], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    image = tf.image.random_flip_left_right(image)
    return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocessing_image(nc_roi, art_roi, pv_roi, nc_patch, art_patch, pv_patch, output_height, output_width, is_training):
    if is_training:
        preprocessing_image_train(nc_roi, art_roi, pv_roi, nc_patch, art_patch, pv_patch, output_height, output_width)