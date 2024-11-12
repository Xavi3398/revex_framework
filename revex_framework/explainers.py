import numpy as np
import sklearn
from sklearn.utils import check_random_state
from functools import partial
from scipy.special import binom
from sklearn.preprocessing import StandardScaler

from revex_framework.lime_base import LimeBase


class Explainer(object):

    def __init__(self, data, labels):
        """ Base class for explaining the prediction of a classifier for a
        video, based on the predictions for a set of perturbed instances
        of the video.

        Neither the video nor the classifier are needed for this task,
        only the generated perturbations of the input (data) and the
        prediction of the classifier for each perturbed sample and class
        (labels).

        Args:
            data (2d numpy array): array of shape: [num_samples+1 (or +2),
                num_features]. This array will have 0 where a region is
                occluded for a specific instance, and 1 where not. The first
                row corresponds to the original video without perturbations.
                In case that AccumPerturber was used, the second row will
                correspond to the fully occluded video.
            labels (2d numpy array): array of shape: [n_samples, n_classes],
                conaining the prediction of the classifier for each class and
                perturbed sample.
        """

        self.data = data
        if data is not None:
            self.num_samples = data.shape[0]
            self.num_features = data.shape[1]
        self.labels = labels

    def explain(self, label_to_explain=None):
        pass


class LimeExplainer(Explainer):

    def __init__(self, data, labels):
        """ Generates LIME explanations, by training a simple regression model
        with the set of classification scores for each perturbed video
        sample and learning the importance of each region. The importance
        will have a value between [-1, 1].

        The code has been adapted from the offical LIME repository:
        https://github.com/marcotcr/lime.

        Args:
            data (2d numpy array): array of shape: [num_samples+1,
                num_features]. This array will have 0 where a region is
                occluded for a specific instance, and 1 where not. The first
                row corresponds to the original video without perturbations.
                In case that AccumPerturber was used, the second row will
                correspond to the fully occluded video.
            labels (2d numpy array): array of shape: [n_samples, n_classes],
                conaining the prediction of the classifier for each class and
                perturbed sample.
        """
        super().__init__(data, labels)

    def explain(self, label_to_explain=None, kernel_width=.25, kernel=None,
                verbose=False, feature_selection='auto', num_features=100000,
                distance_metric='cosine', model_regressor=None,
                random_state=None, non_perturbed_samples=None):
        """ Explains the importance of the different regions for a target
        class.

        Args:
            label_to_explain (int, optional): label to explain. If None,
                the class with the highest score for the unperturbed video is
                chosen as label to explain. Defaults to None.
            kernel_width (float, optional): kernel width for the exponential
                kernel. Defaults to .25.
            kernel (function, optional): similarity kernel that takes euclidean
                distances and kernel width as input and outputs weights in
                (0,1). If None, defaults to an exponential kernel. Defaults to
                None.
            verbose (bool, optional): whether to print local prediction values
                from linear model or not. Defaults to False.
            feature_selection (str, optional): feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does. Defaults to 'auto'.
            num_features (int, optional): maximum number of features present in
                explanation. Defaults to 100000.
            distance_metric (str, optional): the distance metric to use for
                weights. Defaults to 'cosine'.
            model_regressor (sklearn regressor, optional): sklearn regressor to
                use in explanation. Must have model_regressor.coef_ and
                'sample_weight' as a parameter to model_regressor.fit(). If
                None, Ridge regression is used. Defaults to None.
            random_state (int, optional): seed for random number generator.
                Defaults to None.
            non_perturbed_samples (list, optional): list of indices of samples
                that were not perturbed. Only necessary when estimating Shapley
                values. Defaults to None.

        Returns:
            scores (list): a list of pairs [id_region, score], where id_region 
                is in the range [0, n_regions] and the score is in the range 
                [-1.0, 1.0].
        """

        # Label to explain
        if label_to_explain is None:
            label_to_explain = np.argmax(self.labels[0])

        # Prepare LimeBase
        if kernel is None or kernel == 'lime':
            kernel_fn = partial(lime_kernel, kernel_width=float(kernel_width),
                                distance_metric=distance_metric)
        elif kernel == 'shap':
            kernel_fn = shap_kernel
        else:
            kernel_fn = kernel
        base = LimeBase(
            kernel_fn,
            verbose,
            random_state=check_random_state(random_state))

        # Remove non perturbed samples when training simple model
        # Necessary when estimating Shapley values
        if non_perturbed_samples is not None:
            perturbed_samples = \
                [i for i in list(range(self.data.shape[0]))
                 if i not in non_perturbed_samples]
            data_perturbed = self.data[perturbed_samples, :]
            labels_perturbed = self.labels[perturbed_samples, :]
        else:
            data_perturbed = self.data
            labels_perturbed = self.labels

        # Simple model fitting
        return base.explain_instance_with_data(
            data_perturbed, labels_perturbed, label_to_explain, num_features,
            model_regressor=model_regressor,
            feature_selection=feature_selection)[1]


class MeanExplainer(Explainer):

    def __init__(self, data, labels):
        """ Generates Mean explanations, by averaging the confidence of the
        predictions on perturbed videos for each region and class.
        The importance will have a value between [0.0, 1.0].

        Args:
            data (2d numpy array): array of shape: [num_samples+1,
                num_features]. This array will have 0 where a region is
                occluded for a specific instance, and 1 where not. The first
                row corresponds to the original video without perturbations.
                In case that AccumPerturber was used, the second row will
                correspond to the fully occluded video.
            labels (2d numpy array): array of shape: [n_samples, n_classes],
                conaining the prediction of the classifier for each class and
                perturbed sample.
        """
        super().__init__(data, labels)

    def explain(self, label_to_explain=None):
        """ Explains the importance of the different regions for a target
        class.

        Args:
            label_to_explain (int, optional): label to explain. If None,
                the class with the highest score for the unperturbed video is
                chosen as label to explain. Defaults to None.

        Returns:
            scores (list): a list of pairs [id_region, score], where id_region 
                is in the range [0, n_regions] and the score is in the range 
                [-1.0, 1.0].
        """

        # Label to explain
        if label_to_explain is None:
            label_to_explain = np.argmax(self.labels[0])

        # Score for each feature of each sample, where it was not occluded
        scores = np.zeros_like(self.data, dtype=float)
        for i in range(self.num_samples):
            scores[i, :] = self.labels[i, label_to_explain] * self.data[i, :]

        # Mean of each feature over samples
        scores_regions = np.zeros(shape=(self.num_features), dtype=float)
        for i in range(self.num_features):
            scores_regions[i] = np.mean(scores[:, i][self.data[:, i] != 0])

        # Normalization
        scores_regions = scores_regions / np.sum(scores_regions)

        return list(zip(range(scores_regions.shape[0]), scores_regions))


class ShapExplainer(Explainer):

    def __init__(self, data, labels):
        """ Generates SHAP explanations, by averaging, for each feature, the
        difference in the confidence of a prediction when "unoccluding" it.
        The importance will have a value between [-1.0, 1.0].

        This Explainer requires using the AccumPerturber to generate the
        perturbations, since regions have to be added to the video one by
        one.

        Args:
            data (2d numpy array): array of shape: [num_samples+2,
                num_features]. This array will have 0 where a region is
                occluded for a specific instance, and 1 where not. The first
                row corresponds to the original video without perturbations.
                The second row will correspond to the fully occluded video.
            labels (2d numpy array): array of shape: [n_samples, n_classes],
                conaining the prediction of the classifier for each class and
                perturbed sample.
        """
        super().__init__(data, labels)

    def explain(self, label_to_explain=None):
        """ Explains the importance of the different regions for a target
        class.

        Args:
            label_to_explain (int, optional): label to explain. If None,
                the class with the highest score for the unperturbed video is
                chosen as label to explain. Defaults to None.

        Returns:
            scores (list): a list of pairs [id_region, score], where id_region 
                is in the range [0, n_regions] and the score is in the range 
                [-1.0, 1.0].
        """

        # Label to explain
        if label_to_explain is None:
            label_to_explain = np.argmax(self.labels[0])

        # Empty prediction, where no features are present
        zero_score = self.labels[1, label_to_explain]

        # List of scores for each feature
        scores = [[] for _ in range(self.num_features)]
        for i in range(self.num_samples-2):

            # If first feature, we use the empty prediction as the last score
            if i % self.num_features == 0:
                last_data = self.data[1, :]
                last_score = zero_score

            # Get data and score of new feature
            new_data = self.data[i+2, :]
            new_score = self.labels[i + 2, label_to_explain]

            # Find number of feature and store difference with last score
            new_feature = np.argmax(np.subtract(new_data, last_data))
            scores[new_feature].append(new_score - last_score)

            # Update last_score
            last_score = new_score
            last_data = new_data

        # Compute average between samples for each feature and return scores
        return [[i, sum(s)/len(s)] for i, s in enumerate(scores)]


class DifferenceExplainer(Explainer):

    def __init__(self, data, labels):
        """ Generates Difference explanations, by computing difference between
        base case (original video) prediction confidence and single
        perturbations (where only one region is occluded at once).
        The importance will have a value between [-1.0, 1.0].

        This Explainer requires using the SinglePerturber to generate the
        perturbations, since num_samples=num_features+1 (includes the base
        case), where only one occlusion per sample is present.

        Args:
            data (2d numpy array): array of shape: [num_samples+1,
                num_features]. This array will have 0 where a region is
                occluded for a specific instance, and 1 where not. The first
                row corresponds to the original video without perturbations.
                In case that AccumPerturber was used, the second row will
                correspond to the fully occluded video.
            labels (2d numpy array): array of shape: [n_samples, n_classes],
                conaining the prediction of the classifier for each class and
                perturbed sample.
        """
        super().__init__(data, labels)

    def explain(self, label_to_explain=None, invert=False):
        """ Explains the importance of the different regions for a target
        class.

        Args:
            label_to_explain (int, optional): label to explain. If None,
                the class with the highest score for the unperturbed video is
                chosen as label to explain. Defaults to None.
            invert (bool, optional): whether to invert the values when
                computing the difference. False should be used when labels
                contains the relevance of removed regions (e.g. Single
                Perturbations), while True should be used when labels contains
                the relevance of non removed regions (e.g. AllButOne
                perturbations).

        Returns:
            scores (list): a list of pairs [id_region, score], where id_region 
                is in the range [0, n_regions] and the score is in the range 
                [-1.0, 1.0].
        """

        # Label to explain
        if label_to_explain is None:
            label_to_explain = np.argmax(self.labels[0])

        # Score for each feature of each sample, where it was not occluded
        if not invert:
            scores_regions = self.labels[0, label_to_explain] \
                - self.labels[1:, label_to_explain]
        else:
            scores_regions = self.labels[1:, label_to_explain] \
                - self.labels[0, label_to_explain]

        return list(zip(range(scores_regions.shape[0]), scores_regions))


class ValueExplainer(Explainer):

    def __init__(self, data, labels):
        """ Generates Value explanations, by directly using prediction
        confidence of single perturbations (where only one region is
        occluded at once) as explanations.
        The importance will have a value between [0.0, 1.0].

        This Explainer requires using the SinglePerturber to generate the
        perturbations, since num_samples=num_features+1 (includes the base
        case), where only one occlusion per sample is present.

        Args:
            data (2d numpy array): array of shape: [num_samples+1,
                num_features]. This array will have 0 where a region is
                occluded for a specific instance, and 1 where not. The first
                row corresponds to the original video without perturbations.
                In case that AccumPerturber was used, the second row will
                correspond to the fully occluded video.
            labels (2d numpy array): array of shape: [n_samples, n_classes],
                conaining the prediction of the classifier for each class and
                perturbed sample.
        """
        super().__init__(data, labels)

    def explain(self, label_to_explain=None):
        """ Explains the importance of the different regions for a target
        class.

        Args:
            label_to_explain (int, optional): label to explain. If None,
                the class with the highest score for the unperturbed video is
                chosen as label to explain. Defaults to None.

        Returns:
            scores (list): a list of pairs [id_region, score], where id_region 
                is in the range [0, n_regions] and the score is in the range 
                [-1.0, 1.0].
        """

        # Label to explain
        if label_to_explain is None:
            label_to_explain = np.argmax(self.labels[0])

        # Score for each feature of each sample, where it was not occluded
        scores_regions = self.labels[1:, label_to_explain]

        return list(zip(range(scores_regions.shape[0]), scores_regions))


def lime_kernel(data, kernel_width=0.25, distance_metric='cosine'):
    """ Kernel used by LIME for computing the weight of each sample (locality).
    The exponential kernel is used on a distance function which can be
    chosen.

    Args:
        data (2d numpy array): array of shape: [num_samples+1 (+2),
            num_features]. This array will have 0 where a region is
            occluded for a specific instance, and 1 where not. The first
            row corresponds to the original video without perturbations.
            In case that AccumPerturber was used, the second row will
            correspond to the fully occluded video.
        kernel_width (float, optional): kernel width to use for the exponential
            kernel. Defaults to 0.25.
        distance_metric (str, optional): distance metric to use for the
            sklearn.metrics.pairwise_distances function. See its documentation
            for all possible values. Defaults to 'cosine'.

    Returns:
        weights (1d numpy array): array of shape: [num_samples], with the
            weight to use for each sample.
    """
    distances = sklearn.metrics.pairwise_distances(
            data,
            np.ones(shape=(1, data.shape[1]), dtype='int32'),
            metric=distance_metric
        ).ravel()
    return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))


def shap_kernel(data):
    """ Kernel used to estimate Shapley values using LIME. The weighting
    function is changed to match the SHAP paper. In this case, the weight
    of a sample depends on the total of features and the number of occluded
    features.

    Infinity is simulated using a great value when all or any of the
    features are occluded. Avoid these cases by removing non perturbed
    samples from data, since they will affect the weights when training the
    simple model.

    Args:
        data (2d numpy array): array of shape: [num_samples+1 (+2),
            num_features]. This array will have 0 where a region is
            occluded for a specific instance, and 1 where not. The first
            row corresponds to the original video without perturbations.
            In case that AccumPerturber was used, the second row will
            correspond to the fully occluded video.

    Returns:
        weights (1d numpy array): array of shape: [num_samples], with the
            weight to use for each sample.
    """
    m = data.shape[1]  # Number of features
    z = np.sum(data, axis=1)  # Number of non zero features per sample
    result = []
    for zi in z:
        if zi == m or zi == 0:
            result.append(100000000)
        else:
            result.append((m-1)/(binom(m, zi)*zi*(m-zi)))

    return StandardScaler(np.array(result))
