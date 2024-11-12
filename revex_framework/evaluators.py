import numpy as np
import cv2
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, auc
from scipy import ndimage
import copy

from revex_framework.utils import save_video, painter, histogram_stretching, blur_video


class Evaluator(object):

    def __init__(self, video, score_map, classifier_fn, label,
                 invert_importance=False, hide_color=0, blur_mode='3d', 
                 blur_radius=25, blur_sigma=None):
        """ Base for XAI methods evaluators.

        Args:
            video (4d numpy array): video being explained.
            score_map (numpy array): score map with relevance of the pixels of a
                video.
            classifier_fn (function): function that predicts the label of a video.
            label (int): label to explain. If None, the class with the highest 
                score for the unperturbed video is chosen as label to explain. 
                Defaults to None.
            invert_importance (bool, optional): whether to invert the importance
                of the score map. Defaults to False.
            hide_color (int | string, optional): color to use to occlude a region. 
                If None, the mean color of a region thrughout the video is used for 
                that region. If 'blur', the video is blurred using a Gaussian blur
                and used to perturb the input video. Defaults to None.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be '3d' or '2d'. Defaults to '3d'.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.
        """


        self.video = video

        self.score_map = score_map.copy()
        # self.score_map[self.score_map < 0] = 0  # Remove negative relevance
        self.score_map = histogram_stretching(self.score_map)

        # Invert colors: 0->blue, 255->red for default colormap
        if invert_importance:
            self.score_map = 1 - self.score_map

        # Resizing
        if self.video.shape[:3] != self.score_map.shape:
            scale = [self.video.shape[i] / self.score_map.shape[i]
                     for i in range(len(self.score_map.shape))]
            self.score_map = ndimage.zoom(self.score_map, scale, order=1)

        self.classifier_fn = classifier_fn
        self.label = label

        # Init fudged video
        self.fudged_video = video.copy()

        if type(hide_color) == int:
            self.fudged_video[:] = hide_color

        # Use a blurred video to occlude regions
        elif type(hide_color) == str and hide_color == 'blur':
            self.fudged_video = blur_video(
                video, 
                mode=blur_mode, 
                radius=blur_radius, 
                sigma=blur_sigma)
        
        # Use a specified video to occlude regions
        else:
            self.fudged_video = hide_color

    def evaluate(self):
        pass


class AUCEvaluator(Evaluator):

    def __init__(self, video, score_map, classifier_fn, label,
                 method, invert_importance=False, hide_color=0, 
                 blur_mode='3d', blur_radius=25, blur_sigma=None):
        """ Evaluator using the AUC of the deletion and insertion games,
        proposed by Petsiuk et al. (see https://arxiv.org/abs/1806.07421).

        Args:
            video (4d numpy array): video being explained.
            score_map (numpy array): score map with relevance of the pixels of a
                video.
            classifier_fn (function): function that predicts the label of a video.
            label (int): label to explain. If None, the class with the highest 
                score for the unperturbed video is chosen as label to explain. 
                Defaults to None.
            method (string): method to use for the AUC computation. Can be
                'deletion' or 'insertion'.
            invert_importance (bool, optional): whether to invert the importance
                of the score map. Defaults to False.
            hide_color (int | string, optional): color to use to occlude a region. 
                If None, the mean color of a region thrughout the video is used for 
                that region. If 'blur', the video is blurred using a Gaussian blur
                and used to perturb the input video. Defaults to None.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be '3d' or '2d'. Defaults to '3d'.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.
        """
        super().__init__(video=video, score_map=score_map, 
                         classifier_fn=classifier_fn, label=label, 
                         invert_importance=invert_importance,
                         hide_color=hide_color, blur_mode=blur_mode, 
                         blur_radius=blur_radius, blur_sigma=blur_sigma)

        self.method = method

    def perturb_and_predict(self, unique=True, n_samples=100,
                            show_progress=True, save_path=None,
                            target_pred=None):
        """ Perturb the video and predict the class of the perturbed video at
        different thresholds. If method is 'deletion', the perturbations
        are done by removing the most important pixels. If method is
        'insertion', the perturbations are done by including the most
        important pixels. Store the percentage of perturbed pixels and 
        the predictions of the classifier.

        Args:
            unique (bool, optional): whether to sample the score map or to
                find all different values. Defaults to True.
            n_samples (int, optional): number of samples to use. Defaults to 100.
            show_progress (bool, optional): whether to show a progress bar or not.
                Defaults to True.
            save_path (string, optional): path to save the perturbed videos.
                Defaults to None.
            target_pred (float, optional): target prediction to stop the computation.
                If None, the computation is done for all samples. Defaults to None.
        """

        pixels = []
        predictions = []

        total_pixels = self.score_map.size

        # No perturbations or all perturbed
        temp = self.video if self.method == 'deletion' \
            else self.fudged_video
        pixels.append(0)
        predictions.append(self.classifier_fn(
            np.array([temp]))[0, self.label])

        # Save first video
        if save_path is not None:
            save_video(temp, save_path, 'per-0.0000.mp4')

        # Choose between sampling the score map or finding all
        # different values
        if unique:
            values = np.unique(self.score_map)[1:][::-1]
        else:
            values = range(n_samples-1, 0, -1)

        # show progress bar
        progress = tqdm(values) if show_progress else values

        last_pred = None
        last_pixels = None

        for i in progress:

            # Current importance threshold
            th = i if unique else i / n_samples

            # Compute which pixels to delete
            mask = self.score_map >= th if self.method == 'deletion' \
                else self.score_map < th
            remaining_pixels = total_pixels - np.count_nonzero(mask)

            # Value of pixels to store: removed or deleted
            if self.method == 'deletion':
                store_p = (total_pixels-remaining_pixels)/total_pixels
            else:
                store_p = remaining_pixels/total_pixels
            pixels.append(store_p)

            if remaining_pixels == last_pixels:
                predictions.append(last_pred)
            else:
                # Remove regions
                if remaining_pixels/total_pixels > 0.5:
                    temp = self.video.copy()
                    temp[mask, :] = self.fudged_video[mask, :]
                # Include regions
                else:
                    mask = np.logical_not(mask)
                    temp = self.fudged_video.copy()
                    temp[mask, :] = self.video[mask, :]

                # Predict
                pred = self.classifier_fn(np.array([temp]))[0, self.label]
                predictions.append(pred)

                # Update last values
                last_pred = pred
                last_pixels = remaining_pixels

                # Save perturbations
                if save_path is not None:
                    save_video(temp, save_path, f'per-{store_p:.4f}.mp4')

                if target_pred is not None and \
                        ((self.method == 'deletion' and pred < target_pred) or
                         (self.method == 'insertion' and pred > target_pred)):
                    break

        # No perturbations
        temp = self.fudged_video if self.method == 'deletion' \
            else self.video
        pixels.append(1)
        predictions.append(self.classifier_fn(
            np.array([temp]))[0, self.label])

        # Save last video
        if save_path is not None:
            save_video(temp, save_path, 'per-1.mp4')

        self.pixels = np.array(pixels)
        self.predictions = np.array(predictions)

    def compute_auc(self):
        """ Compute the AUC of the deletion or insertion game.
        """
        self.auc_value = auc(self.pixels, self.predictions)

    def evaluate(self, unique=True, n_samples=100, show_progress=True,
                 save_path=None):
        """ Evaluate the AUC of the deletion or insertion game.
        Args:
            unique (bool, optional): whether to sample the score map or to
                find all different values. Defaults to True.
            n_samples (int, optional): number of samples to use. Defaults to 100.
            show_progress (bool, optional): whether to show a progress bar or not.
                Defaults to True.
            save_path (string, optional): path to save the perturbed videos.
                Defaults to None.
        Returns:
            float: AUC value.
        """

        self.perturb_and_predict(
            unique, n_samples, show_progress, save_path)

        self.compute_auc()

        return self.auc_value

    def plot_curve(self, title=None, save_path=None, ax=None, title_size=16,
                   show_axes=True):
        """ Plot the curve of the deletion or insertion game.

        Args:
            title (string, optional): title of the plot. Defaults to None.
            save_path (string, optional): path to save the plot. Defaults to None.
            ax (matplotlib axis, optional): axis to plot the curve. Defaults to None.
            title_size (int, optional): size of the title. Defaults to 16.
            show_axes (bool, optional): whether to show axes or not. Defaults to True.
        """

        if ax is not None:
            plt.axes(ax)
        else:
            plt.figure(figsize=(4, 4))
        plt.fill_between(
            x=np.array(self.pixels)*100,
            y1=np.zeros(len(self.pixels)),
            y2=self.predictions,
            alpha=0.6,
            color='tab:orange' if self.method == 'deletion' else 'tab:blue')
        plt.text(50, 0.5, 'AUC='+str(round(self.auc_value, 3)),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=12, fontweight='bold')

        if show_axes:
            if self.method == 'deletion':
                plt.xlabel('% pixels deleted')
            else:
                plt.xlabel('% pixels inserted')
            plt.ylabel('Model Prediction')
        else:
            # plt.axis('off')
            plt.xticks([])
            plt.yticks([])

        if title is not None:
            plt.title(title, fontdict={'fontsize': title_size})

        if save_path is not None:
            plt.savefig(save_path)
        elif ax is None:
            plt.show()


class MinMaskEvaluator(AUCEvaluator):

    def __init__(self, video, score_map, classifier_fn, label,
                 method, invert_importance=False, hide_color=None, blur_mode='3d', 
                 blur_radius=25, blur_sigma=None):
        """ Evaluator using the deletion or insertion game to find the minimum
        number of pixels that get a target prediction, proposed by Fong et al.
        (https://doi.org/10.1109/ICCV.2017.371).

        Args:
            video (4d numpy array): video being explained.
            score_map (numpy array): score map with relevance of the pixels of a
                video.
            classifier_fn (function): function that predicts the label of a video.
            label (int): label to explain. If None, the class with the highest 
                score for the unperturbed video is chosen as label to explain. 
                Defaults to None.
            method (string): method to use for the AUC computation. Can be
                'deletion' or 'insertion'.
            invert_importance (bool, optional): whether to invert the importance
                of the score map. Defaults to False.
            hide_color (int | string, optional): color to use to occlude a region. 
                If None, the mean color of a region thrughout the video is used for 
                that region. If 'blur', the video is blurred using a Gaussian blur
                and used to perturb the input video. Defaults to None.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be '3d' or '2d'. Defaults to '3d'.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.
        """
        super().__init__(video, score_map, classifier_fn, label,
                         method, invert_importance, hide_color, blur_mode, 
                         blur_radius, blur_sigma)

    def perturb_and_predict_th(self, ths=[.8, .9, .95, .99], unique=True,
                               n_samples=100, show_progress=True,
                               save_path=None):
        """ Perturb the video and predict the class of the perturbed video at
        different thresholds. If method is 'deletion', the perturbations
        are done by removing the most important pixels. If method is
        'insertion', the perturbations are done by including the most
        important pixels. Stores the minimum number of pixels that get each
        target threshold.

        Args:
            ths (list): list of target thresholds.
            unique (bool, optional): whether to sample the score map or to
                find all different values. Defaults to True.
            n_samples (int, optional): number of samples to use. Defaults to 100.
            show_progress (bool, optional): whether to show a progress bar or not.
                Defaults to True.
            save_path (string, optional): path to save the perturbed videos.
                Defaults to None.
        """

        # Stop computation when min/max prediction is reached
        if self.method == 'deletion':
            target_pred = np.min(ths)
        elif self.method == 'insertion':
            target_pred = np.max(ths)

        # Perturb and predict, storing pixels and predictions
        self.perturb_and_predict(
            unique, n_samples, show_progress, save_path,
            target_pred=target_pred)

    def get_min_pixels(self, ths=[.8, .9, .95, .99]):
        """
        Get minimum number of pixels that get each target threshold.

        Args:
            ths (list): list of target thresholds.

        Returns:
            list: minimum number of pixels that get each target threshold.
        """

        results = []

        for th in ths:

            # Minimum number of pixels that when deleted decrease the
            # predictions to a target threshold
            if self.method == 'deletion':
                pixels = self.pixels[self.predictions <= th]
            # Minimum number of pixels that when inserted increase the
            # predictions to a target threshold
            elif self.method == 'insertion':
                pixels = self.pixels[self.predictions >= th]

            # Check if there is at least a solution
            # There is not one when the model does not get a prediction
            # over th on the non-perturbed input video
            if pixels.size > 0:
                results.append(np.min(pixels))
            else:
                results.append(None)

        return results

    def evaluate(self, ths=[.8, .9, .95, .99], unique=True, n_samples=100,
                 show_progress=True, save_path=None):
        """ Evaluate the minimum number of pixels that get each target threshold.

        Args:
            ths (list): list of target thresholds.
            unique (bool, optional): whether to sample the score map or to
                find all different values. Defaults to True.
            n_samples (int, optional): number of samples to use. Defaults to 100.
            show_progress (bool, optional): whether to show a progress bar or not.
                Defaults to True.
            save_path (string, optional): path to save the perturbed videos.
                Defaults to None.

        Returns:
            list: minimum number of pixels that get each target threshold.
        """

        self.perturb_and_predict_th(ths, unique, n_samples, show_progress,
                                    save_path)

        return self.get_min_pixels(ths)


class DropEvaluator(Evaluator):

    def __init__(self, video, score_map, classifier_fn, label, 
                 invert_importance=False, hide_color=None, blur_mode='3d', 
                 blur_radius=25, blur_sigma=None):
        """ Evaluator using the average drop game, proposed by 
        Chattopadhyay et al. (https://doi.org/10.1109/WACV.2018.00097).

        Args:
            video (4d numpy array): video being explained.
            score_map (numpy array): score map with relevance of the pixels of a
                video.
            classifier_fn (function): function that predicts the label of a video.
            label (int): label to explain. If None, the class with the highest 
                score for the unperturbed video is chosen as label to explain. 
                Defaults to None.
            invert_importance (bool, optional): whether to invert the importance
                of the score map. Defaults to False.
            hide_color (int | string, optional): color to use to occlude a region. 
                If None, the mean color of a region thrughout the video is used for 
                that region. If 'blur', the video is blurred using a Gaussian blur
                and used to perturb the input video. Defaults to None.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be '3d' or '2d'. Defaults to '3d'.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.
        """
        super().__init__(video, score_map, classifier_fn, label, 
                         invert_importance, hide_color, blur_mode, 
                         blur_radius, blur_sigma)

    def perturb_and_predict(self, save_path=None):
        """ Perturb the video and predict the class of the perturbed video.

        Args:
            save_path (string, optional): path to save the perturbed video.
                Defaults to None.

        Returns:
            list: predictions for the unperturbed and perturbed videos.
        """

        # Base case
        y = self.classifier_fn(np.array([self.video]))[0, self.label]

        # Perturbed case
        temp = copy.deepcopy(self.fudged_video)
        temp = (self.video * self.score_map[..., None] + 
                (1 - self.score_map)[..., None] * self.fudged_video).astype('uint8')
        o = self.classifier_fn(np.array([temp]))[0, self.label]

        # Save video
        if save_path is not None:
            save_video(temp, save_path)

        return [y, o]

    def evaluate(self, save_path=None):
        """ Evaluate the average drop game.

        Args:
            save_path (string, optional): path to save the perturbed video.
                Defaults to None.

        Returns:
            float: percentage of drop in the prediction.
        """
        y, o = self.perturb_and_predict(save_path)
        return 100 * max(0, (y - o)) / y


class PointingGameEvaluator(Evaluator):

    def __init__(self, video, score_map, method='mask', 
                 invert_importance=False):
        """ Evaluator using the pointing game, proposed by Zhang et al.
        (https://doi.org/10.1007/978-3-319-46493-0_33).

        Args:
            video (4d numpy array): video being explained.
            score_map (numpy array): score map with relevance of the pixels of a
                video.
            method (string, optional): method to use for the evaluation. Can be
                'bb' (bounding box) or 'mask'. Defaults to 'mask'.
            invert_importance (bool, optional): whether to invert the importance
                of the score map. Defaults to False.
        """
            
        super().__init__(video, score_map, None, None, invert_importance)
        self.method = method
        if method == 'bb':
            self.evaluate = self.evaluate_bb
        elif method == 'mask':
            self.evaluate = self.evaluate_mask

    def evaluate_bb(self, bb, save_path=None):
        """ Evaluate the pointing game using a bounding box.

        Args:
            bb (list): list containing a tuple of starting and ending 
                coordinates in each dimension (3 dimensions for video).
            save_path (string, optional): path to save the video with the most
                important point. Defaults to None.

        Returns:
            in_bb (bool): whether the most important point is inside the bounding box.
        """

        # Get coordinates of points with maximum importance
        expl_coords = np.nonzero(self.score_map == np.max(self.score_map))

        # Check if the mean coordinate of most important points is in the
        # bounding box, in each dimension
        for dim in range(len(self.score_map.shape)):
            pp_coord = np.mean(expl_coords[dim])
            if pp_coord < bb[dim][0] or pp_coord > bb[dim][1]:
                return False

        if save_path is not None:
            expl_coords2 = tuple([round(np.mean(expl_coords[dim])) 
                                  for dim in range(len(self.score_map.shape))])
            mask = bb_to_mask(bb, self.score_map.shape)

            temp = np.zeros_like(self.video)
            temp[mask == 1,:] = [0, 255, 0]	
            temp2 = painter(self.video, temp, alpha2=0.75)
            temp2[temp == 0] = self.video[temp == 0]
            cv2.circle(temp2[expl_coords2[0],...], 
                       [expl_coords2[2], expl_coords2[1]], 5, (255, 0, 0), -1)
            save_video(temp2, save_path)

        return True

    def evaluate_mask(self, mask, save_path=None):
        """ Evaluate the pointing game using a mask.

        Args:
            mask (numpy array): mask with the region of interest.
            save_path (string, optional): path to save the video with the most
                important point. Defaults to None.

        Returns:
            in_mask (bool): whether the most important point is inside the mask.
        """

        # Get coordinates of points with maximum importance
        expl_coords = np.nonzero(self.score_map == np.max(self.score_map))

        # Get mean coordinate of most important points
        expl_coords2 = tuple([int(round(np.mean(expl_coords[dim]))) for dim in range(len(self.score_map.shape))])

        if save_path is not None:
            temp = np.zeros_like(self.video)
            temp[mask == 1,:] = [0, 255, 0]	
            temp2 = painter(self.video, temp, alpha2=0.75)
            temp2[mask == 0] = self.video[mask == 0]
            cv2.circle(temp2[expl_coords2[0],...], [expl_coords2[2], expl_coords2[1]], 5, (255, 0, 0), -1)
            save_video(temp2, save_path)

        # Check if the mean coordinate of the most important point is in the mask
        return mask[expl_coords2] == 1


class IOUEvaluator(Evaluator):

    def __init__(self, video, score_map, method='mask', invert_importance=False):
        """ Evaluator using the Intersection Over Union (IOU) metric.

        Args:
            video (4d numpy array): video being explained.
            score_map (numpy array): score map with relevance of the pixels of a
                video.
            method (string, optional): method to use for the evaluation. Can be
                'bb' (bounding box) or 'mask'. Defaults to 'mask'.
            invert_importance (bool, optional): whether to invert the importance
                of the score map. Defaults to False.
        """
        super().__init__(video, score_map, None, None, invert_importance)
        self.method = method
        if method == 'bb':
            self.evaluate = self.evaluate_bb
        elif method == 'mask':
            self.evaluate = self.evaluate_mask

    def evaluate_iou(self, truth_map, th_importance=0.9, th_iou=None, 
                     to_bb=False, save_path=None, use_opening=None):
        """ Evaluate the IOU metric.

        Args:
            truth_map (numpy array): ground truth mask.
            th_importance (float, optional): importance threshold to use for
                the score_map. Defaults to 0.9.
            th_iou (float, optional): IOU threshold. If None, the IOU value is
                returned. Defaults to None.
            to_bb (bool, optional): whether to convert the mask to a bounding box
                or not. Defaults to False.
            save_path (string, optional): path to save the video with the TP, FP
                and FN. Defaults to None.
            use_opening (bool, optional): whether to use morphological opening to
                remove noise or not. Defaults to None.

        Returns:
           iou (float | bool): IOU value if th_iou is not None, whether the IOU value
                is above the threshold otherwise.
        """

        # Score map from importance threshold
        score_map = np.zeros(shape=self.score_map.shape, dtype=int)
        score_map[self.score_map > th_importance] = 1

        # Open mask to remove noise
        if use_opening:
            score_map = cv2.morphologyEx((score_map*255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5,5),np.uint8)) / 255

        # Convert score map mask to bounding box
        if to_bb:
            score_map2 = score_map
            score_map = np.zeros_like(score_map)
            
            for frame in range(score_map2.shape[0]):
                score_map_coords = np.nonzero(score_map2[frame,...] == 1)
                if len(score_map_coords[0]) == 0:
                    continue
                top = np.min(score_map_coords[0])
                bottom = np.max(score_map_coords[0])
                left = np.min(score_map_coords[1])
                right = np.max(score_map_coords[1])

                score_map[frame, top:bottom, left:right] = 1

        # Save video with TP, FP and FN
        if save_path is not None:
            true_positives = np.logical_and(truth_map, score_map)
            false_positives = np.logical_and(np.logical_not(truth_map), score_map)
            false_negatives = np.logical_and(truth_map, np.logical_not(score_map))

            temp = np.zeros_like(self.video)
            temp[true_positives == 1,:] = [0, 0, 255]
            temp[false_positives == 1,:] = [255, 0, 0]
            temp[false_negatives == 1,:] = [0, 255, 00]

            temp2 = painter(self.video, temp, alpha2=.75)
            temp2[np.logical_or(truth_map, score_map) == 0] = self.video[
                np.logical_or(truth_map, score_map) == 0]
            save_video(temp2, save_path)

        # IOU
        iou = jaccard_score(truth_map.flatten(), score_map.flatten())

        # Whether to apply threshold for IOU value or not
        if th_iou is None:
            return iou
        else:
            return iou >= th_iou

    def evaluate_bb(self, bb, th_importance=0.9, th_iou=None, save_path=None):
        """ Evaluate the IOU metric using a bounding box.

        Args:
            bb (list): list containing a tuple of starting and ending 
                coordinates in each dimension (3 dimensions for video)
            th_importance (float, optional): importance threshold to use for
                the score_map. Defaults to 0.9.
            th_iou (float, optional): IOU threshold. If None, the IOU value is
                returned. Defaults to None.
            save_path (string, optional): path to save the video with the TP, FP
                and FN. Defaults to None.

        Returns:
            iou (float | bool): IOU value if th_iou is not None, whether the IOU value
                is above the threshold otherwise
        """

        # Truth map from bounding box
        truth_map = bb_to_mask(bb, self.score_map.shape)

        return self.evaluate_iou(truth_map, th_importance, 
                                 th_iou, save_path=save_path)

    def evaluate_mask(self, mask, th_importance=0.9, th_iou=None, 
                      save_path=None, to_bb=False):
        """ Evaluate the IOU metric using a mask.

        Args:
            mask (numpy array): mask with the region of interest.
            th_importance (float, optional): importance threshold to use for
                the score_map. Defaults to 0.9.
            th_iou (float, optional): IOU threshold. If None, the IOU value is
                returned. Defaults to None.
            save_path (string, optional): path to save the video with the TP, FP
                and FN. Defaults to None.
            to_bb (bool, optional): whether to convert the mask to a bounding box
                or not. Defaults to False.

        Returns:
            iou (float | bool): IOU value if th_iou is not None, whether the IOU value
                is above the threshold otherwise.
        """
        return self.evaluate_iou(mask, th_importance, th_iou, 
                                 save_path=save_path, to_bb=to_bb)


class AUCEvaluatorSimple(AUCEvaluator):

    def __init__(self, pixels, predictions, method):
        """ Simple version of the AUC evaluator, provided for convenience
        when loading from a file.

        Args:
            pixels (numpy array): percentage of perturbed pixels.
            predictions (numpy array): predictions of the classifier.
            method (string): method to use for the AUC computation. Can be
                'deletion' or 'insertion'.
        """
        self.pixels = pixels
        self.predictions = predictions
        self.method = method


class MinMaskEvaluatorSimple(MinMaskEvaluator):

    def __init__(self, pixels, predictions, method):
        """ Simple version of the MinMask evaluator, provided for convenience
        when loading from a file.
        
        Args:
            pixels (numpy array): percentage of perturbed pixels.
            predictions (numpy array): predictions of the classifier.
            method (string): method to use for the AUC computation. Can be
                'deletion' or 'insertion'.
        """
        self.pixels = pixels
        self.predictions = predictions
        self.method = method

    
def bb_to_mask(bb, shape):
    """ Convert bounding box to mask.

    Args:
        bb (list): list containing a tuple of starting and ending 
            coordinates in each dimension (3 dimensions for video).
        shape (tuple): shape of the mask.

    Returns:
        mask (numpy array): mask of the shape of the video, without the
            channels.
    """
    mask = np.zeros(shape=shape, dtype='uint8')
    mask[bb[0][0]:bb[0][1], bb[1][0]:bb[1][1], bb[2][0]:bb[2][1]] = 1
    return mask