# import lime
# from lime import lime_image

"""
Functions for explaining classifiers that use Image data.
"""
import copy
from functools import partial
import pyximport
pyximport.install(setup_args={"script_args" : ["--verbose"]})
import numpy as np
import sklearn
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
from tqdm.auto import tqdm

from lime import lime_base
from lime.wrappers.scikit_image import SegmentationAlgorithm

from skimage.filters import gaussian
from skimage2.utils import _supported_float_type
from skimage.color import rgb2lab
from skimage.util import img_as_float
from skimage2._quickshift_cy import _quickshift_cython


class LimeImageExplainer2(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        """Init function.
        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)

    def explain_instance(self, image, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=1, num_features=100000, num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         progress_bar=True):
        """Generates explanations for a prediction.
        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).
        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: If not None, will hide superpixels with this color.
                Otherwise, use the mean pixel color of the image.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: batch size for model predictions
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
            progress_bar: if True, show tqdm progress bar.
        Returns:
            An ImageExplanation object (see lime_image.py) with the corresponding
            explanations.
        """
        
        
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segmentation_fn is None:
            # segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
            #                                         max_dist=200, ratio=0.2,
            #                                         random_seed=random_seed)
            
            segmentation_fn = quickshift(image, kernel_size=4,max_dist=200,convert2lab=False,random_seed=random_seed)
        segments = segmentation_fn

        fudged_image = image[0].copy()
        image = image[0]
 
        if hide_color is None:
            for x in np.unique(segments):
               
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]),
                    np.mean(image[segments == x][:, 3]),
                    np.mean(image[segments == x][:, 4]),
                    np.mean(image[segments == x][:, 5]),
                    np.mean(image[segments == x][:, 6]),
                    np.mean(image[segments == x][:, 7]),
                    np.mean(image[segments == x][:, 8]),
                    np.mean(image[segments == x][:, 9]),
                    np.mean(image[segments == x][:, 10]),
                    np.mean(image[segments == x][:, 11])
                    # np.mean(image[segments == x][:, 12])
                )
        else:
            fudged_image[:] = hide_color

        top = labels

        data, labels = self.data_labels(image, fudged_image, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar)

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10,
                    progress_bar=True):
        """Generates images and predictions in the neighborhood of this image.
        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.
            progress_bar: if True, show tqdm progress bar.
        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features)\
            .reshape((num_samples, n_features))
        labels = []
        data[0, :] = 1
        imgs = []
        rows = tqdm(data) if progress_bar else data
        for row in rows:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
        return data, np.array(labels)
    

    
class ImageExplanation(object):
    def __init__(self, image, segments):
        """Init function.
        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}

    def get_image_and_mask(self, label, positive_only=True, negative_only=False, hide_rest=False,
                           num_features=5, min_weight=0.):
        """Init function.
        Args:
            label: label to explain
            positive_only: if True, only take superpixels that positively contribute to
                the prediction of the label.
            negative_only: if True, only take superpixels that negatively contribute to
                the prediction of the label. If false, and so is positive_only, then both
                negativey and positively contributions will be taken.
                Both can't be True at the same time
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: minimum weight of the superpixels to include in explanation
        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp
                  if x[1] > 0 and x[1] > min_weight][:num_features]
        if negative_only:
            fs = [x[0] for x in exp
                  if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = -1 if w < 0 else 1
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
            return temp, mask
    
    

    
def quickshift(image, ratio=0.2, kernel_size=5, max_dist=10,return_tree=False, sigma=0, 
               convert2lab=True, random_seed=42,*,channel_axis=-1):
    """Segments image using quickshift clustering in Color-(x,y) space.
    Produces an oversegmentation of the image using the quickshift mode-seeking
    algorithm.
    Parameters
    ----------
    image : (width, height, channels) ndarray
        Input image. The axis corresponding to color channels can be specified
        via the `channel_axis` argument.
    ratio : float, optional, between 0 and 1
        Balances color-space proximity and image-space proximity.
        Higher values give more weight to color-space.
    kernel_size : float, optional
        Width of Gaussian kernel used in smoothing the
        sample density. Higher means fewer clusters.
    max_dist : float, optional
        Cut-off point for data distances.
        Higher means fewer clusters.
    return_tree : bool, optional
        Whether to return the full segmentation hierarchy tree and distances.
    sigma : float, optional
        Width for Gaussian smoothing as preprocessing. Zero means no smoothing.
    convert2lab : bool, optional
        Whether the input should be converted to Lab colorspace prior to
        segmentation. For this purpose, the input is assumed to be RGB.
    random_seed : int, optional
        Random seed used for breaking ties.
    channel_axis : int, optional
        The axis of `image` corresponding to color channels. Defaults to the
        last axis.
    Returns
    -------
    segment_mask : (width, height) ndarray
        Integer mask indicating segment labels.
    Notes
    -----
    The authors advocate to convert the image to Lab color space prior to
    segmentation, though this is not strictly necessary. For this to work, the
    image must be given in RGB format.
    References
    ----------
    .. [1] Quick shift and kernel methods for mode seeking,
           Vedaldi, A. and Soatto, S.
           European Conference on Computer Vision, 2008
    """

    image = img_as_float(np.atleast_3d(image))
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    image = image[0]

    if image.ndim > 3:
        raise ValueError("only 2D color images are supported")

    # move channels to last position as expected by the Cython code
    image = np.moveaxis(image, source=channel_axis, destination=-1)

    if convert2lab:
        if image.shape[-1] != 3:
            ValueError("Only RGB images can be converted to Lab space.")
        image = rgb2lab(image)

    if kernel_size < 1:
        raise ValueError("`kernel_size` should be >= 1.")

    image = gaussian(image, [sigma, sigma, 0], mode='reflect', channel_axis=-1)
    image = np.ascontiguousarray(image * ratio)

    segment_mask = _quickshift_cython(
        image, kernel_size=kernel_size, max_dist=max_dist,
        return_tree=return_tree, random_seed=random_seed)
    return segment_mask