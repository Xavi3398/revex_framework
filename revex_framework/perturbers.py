import numpy as np
import os
import copy
from tqdm.auto import tqdm
from sklearn.utils import check_random_state
from scipy import ndimage
from functools import partial
import shap

from revex_framework.utils import save_video, load_video, blur_video


class Perturber(object):

    def __init__(self, video, segments, classifier_fn, hide_color=None,
                 random_state=None, blur_mode='3d', blur_radius=25, blur_sigma=None):
        """ Base class for creating perturbed versions of a video by occluding
        in different ways the regions stablished by a segmentation, and
        obtaining a prediction for each one using the passed function.

        Args:
            video (4d numpy array): video to perturb, of shape:  [n_frames,
                height, width, 3].
            segments (3d numpy array): segmentation of the video, of 
                shape: [n_frames, height, width], where each element indicates the
                region it belongs to.
            classifier_fn (_type_): function to predict the class of a video.
            hide_color (int | string, optional): color to use to occlude a region. 
                If None, the mean color of a region thrughout the video is used for 
                that region. If 'blur', the video is blurred using a Gaussian blur
                and used to perturb the input video. Defaults to None.
            random_state (int, optional): seed to use for replicating
                perturbations. Defaults to None.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be '3d' or '2d'. Defaults to '3d'.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.
        """
        self.video = video
        self.segments = segments
        self.n_features = np.unique(self.segments).shape[0]
        self.classifier_fn = classifier_fn
        self.random_state = check_random_state(random_state)

        # Init fudged video
        self.fudged_video = video.copy()

        # Use mean color for each region
        if hide_color is None:

            # For RISE-like perturbations: the segmentation array is smaller
            # and has to be rescaled. Color of pixels between regions will be
            # the result of interpolation between mean colors of regions.
            if segments.shape != video.shape[:3]:
                scale = [video.shape[i] / segments.shape[i]
                         for i in range(len(segments.shape))]
                segments_big = ndimage.zoom(segments, scale, order=0)
                fudged_small = np.zeros(shape=segments.shape+(3,),
                                        dtype=video.dtype)
                for x in np.unique(segments):
                    fudged_small[segments == x] = (
                        np.mean(video[segments_big == x][:, 0]),
                        np.mean(video[segments_big == x][:, 1]),
                        np.mean(video[segments_big == x][:, 2]))
                self.fudged_video = ndimage.zoom(fudged_small, scale + [1,],
                                                 order=1)

            # Each region will have the same color: the mean color of
            # that region
            else:
                for x in np.unique(segments):
                    self.fudged_video[segments == x] = (
                        np.mean(video[segments == x][:, 0]),
                        np.mean(video[segments == x][:, 1]),
                        np.mean(video[segments == x][:, 2]))

        # Use a specified color for all regions
        elif type(hide_color) == int:
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

        self.data_fn = self.get_data

    def get_data(self, num_samples):
        pass

    def get_non_perturbed_samples(self, num_samples):
        pass

    def perturb_and_predict(self, data, batch_size=10, progress_bar=True,
                            save_videos_path=None, load_videos_path=None,
                            dont_predict=False, blur_mode=None, 
                            blur_radius=25, blur_sigma=None):
        """ Generates perturbed videos and their predictions. It has been
        adapted from the public LIME repository.

        Args:
            data (2d numpy array): array of shape: [num_samples, num_features]
                with 0 where a region is occluded for a specific instance, and
                1 where not.
            batch_size (int, optional): number of videos to be passed together
                to the prediction function. Defaults to 10.
            progress_bar (bool, optional): Whether to display a progress bar or
                not. Defaults to True.
            save_videos_path (string, optional): path of a folder where
                perturbed instances should be stored. No videos will be stored
                if None. Defaults to None.
            load_videos_path (string, optional): path of the folder from where
                to load saved perturbed instances. Defaults to None.
            dont_predict (bool, optional): whether to call classifier_fn or
                not. Useful if only perturbations are wanted, or want to be
                stored. Defaults to False.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be None (no blur applied), '3d' or '2d'. Defaults to None.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.
        Returns:
            labels (2d numpy array): with shape [num_samples, num_classes], 
                conaining the prediction of classifier_fn for each class and 
                perturbed sample.
        """

        self.data = data
        labels = []
        vids = []
        rows = tqdm(data) if progress_bar else data

        # Compute first the mask for each region
        region_mask = np.zeros(shape=(self.n_features,) + self.segments.shape,
                               dtype=bool)
        for i in range(self.n_features):
            region_mask[i, ...] = self.segments == i

        for n_sample, row in enumerate(rows):

            # Generate perturbed video
            if load_videos_path is None:

                # Get all segments that should be occluded.
                # If there are more segments to be occluded than not, get all
                # segments that should not be occluded instead.
                if np.sum(row) > self.n_features/2 or \
                        self.segments.shape != self.video.shape[:3] or \
                        blur_mode is not None:
                    occlude = True
                    zeros = np.where(row == 0)[0]

                    # Initialize perturbed video as a simple copy
                    temp = copy.deepcopy(self.video)
                else:
                    occlude = False
                    zeros = np.where(row == 1)[0]

                    # Initialize perturbed video as all occluded
                    temp = copy.deepcopy(self.fudged_video)

                # If at least one region should be occluded
                if len(zeros > 0):

                    # Join masks of all regions that should be occluded using
                    # logical OR
                    mask = region_mask[zeros[0], ...]
                    for j in zeros[1:]:
                        mask = np.logical_or(mask, region_mask[j, ...])

                    # If mask shape is smaller, a rescale is needed. Used for
                    # RISE-like perturbations. The mask will be of floats, using
                    # interpolation between occluded and not occluded regions.
                    if mask.shape != temp.shape[:3]:
                        scale = [temp.shape[i] / mask.shape[i]
                                 for i in range(len(mask.shape))]
                        mask = ndimage.zoom(mask.astype(float),
                                            scale, order=1)
                        temp = (temp * (1 - mask[...,None]) + self.fudged_video * 
                                mask[...,None]).astype('uint8')

                    # Binary mask (LIME-like), where all occluded regions are
                    # set to the color of that regions in the fudged video
                    
                    # Apply a blur to the mask
                    elif blur_mode is not None:
                        mask = blur_video(mask.astype('float32'), mode=blur_mode, 
                                          radius=blur_radius, sigma=blur_sigma)
                        temp = (temp * (1 - mask[...,None]) + self.fudged_video * 
                                mask[...,None]).astype('uint8')
                    # Add fudged_video regions to the video
                    elif occlude:
                        temp[mask, :] = self.fudged_video[mask, :]
                    # Add original video regions to the video (which is initialized as the
                    # fudged_video)
                    else:
                        temp[mask, :] = self.video[mask, :]
                            
                    # Save perturbed video if specified
                    if save_videos_path is not None:
                        save_video(temp, save_videos_path, str(n_sample)+'.m4v')

            # Load perturbed video from memory
            else:
                temp = load_video(os.path.join(load_videos_path,
                                               str(n_sample)+'.m4v'))

            # Add to the list of videos to predict if not dont_predict
            if not dont_predict:
                vids.append(temp)

            # Predict all videos in list when the batch_size is full. Append
            # results to list
            if len(vids) == batch_size and not dont_predict:
                preds = self.classifier_fn(np.array(vids))
                labels.extend(preds)
                vids = []

        # Predict remaining videos in list. Append results to list
        if len(vids) > 0 and not dont_predict:
            preds = self.classifier_fn(np.array(vids))
            labels.extend(preds)

        self.labels = np.array(labels)

        return np.array(labels)


class MultiplePerturber(Perturber):

    def __init__(self, video, segments, classifier_fn, hide_color=None,
                 random_state=None, blur_mode='3d', blur_radius=25, blur_sigma=None):
        """ Perturber that occludes multiple regions for each perturbed video
        (approximately half of the regions). The number of perturbed videos
        can be chosen with num_samples. The original video without occlusions 
        is appended to the start.

        Args:
            video (4d numpy array): video to perturb, of shape:  [n_frames,
                height, width, 3].
            segments (3d numpy array): segmentation of the video, of shape:
                [n_frames, height, width], where each element indicates the
                region it belongs to.
            classifier_fn (_type_): function to predict the class of a video
            hide_color (int | string, optional): color to use to occlude a region. 
                If None, the mean color of a region thrughout the video is used for 
                that region. If 'blur', the video is blurred using a Gaussian blur
                and used to perturb the input video. Defaults to None.
            random_state (int, optional): seed to use for replicating
                perturbations. Defaults to None.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be '3d' or '2d'. Defaults to '3d'.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.
        """
        super().__init__(video, segments, classifier_fn, hide_color,
                         random_state, blur_mode=blur_mode, 
                         blur_radius=blur_radius, blur_sigma=blur_sigma)

    def perturb(self, num_samples=500, exact=False, p=0.5, batch_size=10,
                progress_bar=True, save_videos_path=None,
                load_videos_path=None, dont_predict=False, blur_mode=None, 
                blur_radius=25, blur_sigma=None):
        """ Generates videos with multiple perturbations and their predictions.

        Args:
            num_samples (int, optional): number of desired perturbed videos.
                Defaults to 500.
            exact (bool, optional): whether to perturb exactly (True) or
                approximately (False) a (1-p) proportion of the regions.
                Defaults to False.
            p (float, optional): probability for each region to be occluded.
                Defaults to 0.5.
            batch_size (int, optional): number of videos to be passed together
                to the prediction function. Defaults to 10.
            progress_bar (bool, optional): Whether to display a progress bar or
                not. Defaults to True.
            save_videos_path (string, optional): path of a folder where
                perturbed instances should be stored. No videos will be stored
                if None. Defaults to None.
            load_videos_path (string, optional): path of the folder from where
                to load saved perturbed instances. Defaults to None.
            dont_predict (bool, optional): whether to call classifier_fn or
                not. Useful if only perturbations are wanted, or want to be
                stored. Defaults to False.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be None (no blur applied), '3d' or '2d'. Defaults to None.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.

        Returns:
            labels (2d numpy array): with shape [num_samples, num_classes], 
                conaining the prediction of classifier_fn for each class 
                and perturbed sample.
        """

        # Create data according to perturbation type
        data = self.get_data(num_samples, exact, p)

        # Perturb videos accordingly and run inference on them
        return data, self.perturb_and_predict(
            data, batch_size, progress_bar, save_videos_path,
            load_videos_path, dont_predict, blur_mode=blur_mode, 
            blur_radius=blur_radius, blur_sigma=blur_sigma)

    def get_data(self, num_samples=500, exact=False, p=0.5):
        """ Occludes multiple regions for each perturbed video.

        Args:
            num_samples (int, optional): number of desired perturbed videos.
                Defaults to 500.
            exact (bool, optional): whether to perturb exactly (True) or
                approximately (False) a (1-p) proportion of the regions.
                Defaults to False.
            p (float, optional): probability for each region to be occluded.
                Defaults to 0.5.

        Returns:
            data (2d numpy array): with shape: [num_samples+1, num_features]. 
                This array will have 0 where a region is occluded for a specific
                instance, and 1 where not. The first row corresponds to the
                original video without perturbations.
        """

        # Shuffle feature indexes and occlude the first (1 - p) of them
        if exact:
            data = np.zeros(shape=(num_samples, self.n_features), dtype=int)
            for i in range(num_samples):
                regions = np.array(range(self.n_features))
                np.random.shuffle(regions)
                data[i, regions[:int(p*len(regions))]] = 1

        # Randomly set each region of each sample to 0 or 1
        else:
            data = np.zeros(shape=(num_samples, self.n_features), dtype=int)
            r_data = np.random.random(num_samples * self.n_features) \
                .reshape((num_samples, self.n_features))
            data[r_data >= p] = 0
            data[r_data < p] = 1

        # Add original video at the beginning
        data = np.vstack([np.ones(self.n_features, dtype=int), data])
        return data

    def get_non_perturbed_samples(self, num_samples=None):
        """ Computes which of the rows in the returned data correspond to
        non-perturbed samples.

        Args:
            num_samples (int): number of desired samples when calling to
                get_data.

        Returns:
            non_perturbed_samples (list): list with indexs of the non-perturbed 
                samples.
        """
        return [0]


class SinglePerturber(Perturber):

    def __init__(self, video, segments, classifier_fn, hide_color=None,
                 random_state=None, blur_mode='3d', blur_radius=25, blur_sigma=None):
        """ Perturber which occludes only one region for each perturbed video.
        The number of samples will be equal to the number of regions in the
        segmentation. 
        
        Useful for computing the difference between occluding or not
        occluding each of the regions (ablation).

        The original video without occlusions is appended to the start.

        Args:
            video (4d numpy array): video to perturb, of shape:  [n_frames,
                height, width, 3].
            segments (3d numpy array): segmentation of the video, of shape:
                [n_frames, height, width], where each element indicates the
                region it belongs to.
            classifier_fn (_type_): function to predict the class of a video.
            hide_color (int | string, optional): color to use to occlude a region. 
                If None, the mean color of a region thrughout the video is used for 
                that region. If 'blur', the video is blurred using a Gaussian blur
                and used to perturb the input video. Defaults to None.
            random_state (int, optional): seed to use for replicating
                perturbations. Defaults to None.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be '3d' or '2d'. Defaults to '3d'.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.
        """
        super().__init__(video, segments, classifier_fn, hide_color,
                         random_state, blur_mode=blur_mode, 
                         blur_radius=blur_radius, blur_sigma=blur_sigma)

    def perturb(self, batch_size=10, progress_bar=True, save_videos_path=None,
                load_videos_path=None, dont_predict=False, blur_mode=None, 
                blur_radius=25, blur_sigma=None):
        """ Generates videos with a single perturbation and their predictions.

        Args:
            batch_size (int, optional): number of videos to be passed together
                to the prediction function. Defaults to 10.
            progress_bar (bool, optional): Whether to display a progress bar or
                not. Defaults to True.
            save_videos_path (string, optional): path of a folder where
                perturbed instances should be stored. No videos will be stored
                if None. Defaults to None.
            load_videos_path (string, optional): path of the folder from where
                to load saved perturbed instances. Defaults to None.
            dont_predict (bool, optional): whether to call classifier_fn or
                not. Useful if only perturbations are wanted, or want to be
                stored. Defaults to False.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be None (no blur applied), '3d' or '2d'. Defaults to None.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.

        Returns:
            labels (2d numpy array): with shape [num_samples, num_classes], 
                conaining the prediction of classifier_fn for each class 
                and perturbed sample.
        """
        # Create data according to perturbation type
        data = self.get_data()

        # Perturb videos accordingly and run inference on them
        return data, self.perturb_and_predict(
            data, batch_size, progress_bar, save_videos_path,
            load_videos_path, dont_predict, blur_mode=blur_mode, 
            blur_radius=blur_radius, blur_sigma=blur_sigma)

    def get_data(self):
        """ Occludes only one region for each perturbed video. Useful for
        computing the difference between occluding or not occluding each of
        the regions. The number of samples will be equal to the number of
        regions in the segmentation.

        Returns:
            data (2d numpy array): with shape: [num_samples+1, num_features]. This
                array will have 0 where a region is occluded for a specific
                instance, and 1 where not. The first row corresponds to the
                original video without perturbations.
        """

        # Use identity matrix with logical not to generate data
        data = np.logical_not(np.eye(self.n_features, self.n_features,
                                     dtype=bool)).astype(int)

        # Add original video at the beginning
        data = np.vstack([np.ones(self.n_features, dtype=int), data])
        return data

    def get_non_perturbed_samples(self, num_samples=None):
        """ Computes which of the rows in the returned data correspond to
        non-perturbed samples.

        Args:
            num_samples (int): number of desired samples when calling to
                get_data.

        Returns:
            non_perturbed_samples (list): list with indexs of the non-perturbed 
                samples.
        """
        return [0]


class AllButOnePerturber(Perturber):

    def __init__(self, video, segments, classifier_fn, hide_color=None,
                 random_state=None, blur_mode='3d', blur_radius=25, blur_sigma=None):
        """ Perturber which occludes all regions but one for each perturbed
        video. The number of samples will be equal to the number of regions
        in the segmentation.

        Useful for computing the difference between occluding or not
        occluding each of the regions.

        The original video without occlusions is appended to the start.

        Args:
            video (4d numpy array): video to perturb, of shape:  [n_frames,
                height, width, 3].
            segments (3d numpy array): segmentation of the video, of shape:
                [n_frames, height, width], where each element indicates the
                region it belongs to.
            classifier_fn (_type_): function to predict the class of a video
            hide_color (int | string, optional): color to use to occlude a region. 
                If None, the mean color of a region thrughout the video is used for 
                that region. If 'blur', the video is blurred using a Gaussian blur
                and used to perturb the input video. Defaults to None.
            random_state (int, optional): seed to use for replicating
                perturbations. Defaults to None.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be '3d' or '2d'. Defaults to '3d'.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.
        """
        super().__init__(video, segments, classifier_fn, hide_color,
                         random_state, blur_mode=blur_mode, 
                         blur_radius=blur_radius, blur_sigma=blur_sigma)

    def perturb(self, batch_size=10, progress_bar=True, save_videos_path=None,
                load_videos_path=None, dont_predict=False, blur_mode=None, 
                blur_radius=25, blur_sigma=None):
        """ Generates videos with a single perturbation and their predictions.

        Args:
            batch_size (int, optional): number of videos to be passed together
                to the prediction function. Defaults to 10.
            progress_bar (bool, optional): Whether to display a progress bar or
                not. Defaults to True.
            save_videos_path (string, optional): path of a folder where
                perturbed instances should be stored. No videos will be stored
                if None. Defaults to None.
            load_videos_path (string, optional): path of the folder from where
                to load saved perturbed instances. Defaults to None.
            dont_predict (bool, optional): whether to call classifier_fn or
                not. Useful if only perturbations are wanted, or want to be
                stored. Defaults to False.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be None (no blur applied), '3d' or '2d'. Defaults to None.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.

        Returns:
            labels (2d numpy array): with shape [num_samples, num_classes], 
                conaining the prediction of classifier_fn for each class 
                and perturbed sample.
        """
        # Create data according to perturbation type
        data = self.get_data()

        # Perturb videos accordingly and run inference on them
        return data, self.perturb_and_predict(
            batch_size, progress_bar, save_videos_path,
            load_videos_path, dont_predict, blur_mode=blur_mode, 
            blur_radius=blur_radius, blur_sigma=blur_sigma)

    def get_data(self):
        """ Occludes only one region for each perturbed video. Useful for
        computing the difference between occluding or not occluding each of
        the regions. The number of samples will be equal to the number of
        regions in the segmentation.

        Returns:
            data (2d numpy array): with shape: [num_samples+1, num_features]. This
                array will have 0 where a region is occluded for a specific
                instance, and 1 where not. The first row corresponds to the
                original video without perturbations.
        """

        # Use identity matrix with to generate data
        data = np.eye(self.n_features, self.n_features, dtype=bool).astype(int)

        # Add fully perturbed video at the beginning
        data = np.vstack([np.zeros(self.n_features, dtype=int), data])
        return data

    def get_non_perturbed_samples(self, num_samples=None):
        """ Computes which of the rows in the returned data correspond to
        non-perturbed samples.

        Args:
            num_samples (int): number of desired samples when calling to
                get_data.

        Returns:
            non_perturbed_samples (list): list with indexs of the non-perturbed 
                samples.
        """
        return [0]

    def perturb_and_predict(self, batch_size=10, progress_bar=True,
                            save_videos_path=None, load_videos_path=None,
                            dont_predict=False, blur_mode=None, 
                            blur_radius=25, blur_sigma=None):
        """ Generates perturbed videos and their predictions. It has been
        adapted from the public LIME repository. Method supersceded
        because it is faster to write only the non perturbed region
        than perturb all regions but one.

        Args:
            batch_size (int, optional): number of videos to be passed together
                to the prediction function. Defaults to 10.
            progress_bar (bool, optional): Whether to display a progress bar or
                not. Defaults to True.
            save_videos_path (string, optional): path of a folder where
                perturbed instances should be stored. No videos will be stored
                if None. Defaults to None.
            load_videos_path (string, optional): path of the folder from where
                to load saved perturbed instances. Defaults to None.
            dont_predict (bool, optional): whether to call classifier_fn or
                not. Useful if only perturbations are wanted, or want to be
                stored. Defaults to False.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be None (no blur applied), '3d' or '2d'. Defaults to None.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.

        Returns:
            labels (2d numpy array): with shape [num_samples, num_classes], 
                conaining the prediction of classifier_fn for each class 
                and perturbed sample.
        """

        labels = []
        vids = []

        # Progress bar
        if progress_bar:
            progress = tqdm(range(self.n_features+1))
        else:
            progress = range(self.n_features+1)

        # Fully perturbed video
        vids.append(self.fudged_video)

        for n_sample in progress:

            # Generate perturbed video
            if load_videos_path is None:

                # Initialize perturbed video as a simple copy
                temp = copy.deepcopy(self.fudged_video)
                mask = self.segments == n_sample

                # If mask shape is smaller, a rescale is needed. Used for
                # RISE-like perturbations. The mask will be of floats, using
                # interpolation between occluded and not occluded regions.
                if mask.shape != temp.shape[:3]:
                    scale = [temp.shape[i] / mask.shape[i]
                             for i in range(len(mask.shape))]
                    mask = ndimage.zoom(mask.astype(float),
                                        scale, order=1)
                    temp = (temp * (1 - mask[...,None]) + self.video * 
                            mask[...,None]).astype('uint8')
                    
                # Apply a blur to the mask
                elif blur_mode is not None:
                    mask = blur_video(mask.astype('float32'), mode=blur_mode, 
                                      radius=blur_radius, sigma=blur_sigma)
                    temp = (temp * (1 - mask[...,None]) + self.video * 
                            mask[...,None]).astype('uint8')
                    
                # Add original video regions to the video (which is initialized as the
                # fudged_video)
                else:
                    temp[mask, :] = self.video[mask, :]

                # Save perturbed video if specified
                if save_videos_path is not None:
                    save_video(temp, save_videos_path, str(n_sample)+'.m4v')

            # Load perturbed video from memory
            else:
                temp = load_video(os.path.join(load_videos_path,
                                               str(n_sample)+'.m4v'))

            # Add to the list of videos to predict if not dont_predict
            if not dont_predict:
                vids.append(temp)

            # Predict all videos in list when the batch_size is full. Append
            # results to list
            if len(vids) == batch_size and not dont_predict:
                preds = self.classifier_fn(np.array(vids))
                labels.extend(preds)
                vids = []

        # Predict remaining videos in list. Append results to list
        if len(vids) > 0 and not dont_predict:
            preds = self.classifier_fn(np.array(vids))
            labels.extend(preds)

        return np.array(labels)


class AccumPerturber(Perturber):

    def __init__(self, video, segments, classifier_fn, hide_color=None,
                 random_state=None, blur_mode='3d', blur_radius=25, blur_sigma=None):
        """ Perturber which starts with all regions occluded and 'unoccludes'
        one on each iteration, until the original video is obtained. This
        process is repeated num_samples times, so the number of perturbed
        videos obtained is equal to num_features*num_samples.

        Useful for observing the change in prediction confidence when
        adding the different regions (features) one by one (SHAP).

        The original video without occlusions and the completely occluded
        video are appended to the start.

        Args:
            video (4d numpy array): video to perturb, of shape:  [n_frames,
                height, width, 3].
            segments (3d numpy array): segmentation of the video, of shape:
                [n_frames, height, width], where each element indicates the
                region it belongs to.
            classifier_fn (_type_): function to predict the class of a video
            hide_color (int | string, optional): color to use to occlude a region. 
                If None, the mean color of a region thrughout the video is used for 
                that region. If 'blur', the video is blurred using a Gaussian blur
                and used to perturb the input video. Defaults to None.
            random_state (int, optional): seed to use for replicating
                perturbations. Defaults to None.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be '3d' or '2d'. Defaults to '3d'.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.
        """
        super().__init__(video, segments, classifier_fn, hide_color,
                         random_state, blur_mode=blur_mode, 
                         blur_radius=blur_radius, blur_sigma=blur_sigma)

    def perturb(self, num_samples=5, batch_size=10, progress_bar=True,
                save_videos_path=None, load_videos_path=None,
                dont_predict=False, blur_mode=None, blur_radius=25,
                blur_sigma=None):
        """ Generates videos with accumulated perturbations and their
        predictions.

        Args:
            num_samples (int, optional): number of desired perturbed videos.
                Defaults to 500.
            batch_size (int, optional): number of videos to be passed together
                to the prediction function. Defaults to 10.
            progress_bar (bool, optional): Whether to display a progress bar or
                not. Defaults to True.
            save_videos_path (string, optional): path of a folder where
                perturbed instances should be stored. No videos will be stored
                if None. Defaults to None.
            load_videos_path (string, optional): path of the folder from where
                to load saved perturbed instances. Defaults to None.
            dont_predict (bool, optional): whether to call classifier_fn or
                not. Useful if only perturbations are wanted, or want to be
                stored. Defaults to False.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be None (no blur applied), '3d' or '2d'. Defaults to None.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.

        Returns:
            labels (2d numpy array): with shape [num_samples, num_classes], 
                conaining the prediction of classifier_fn for each class 
                and perturbed sample.
        """

        # Create data according to perturbation type
        data = self.get_data(num_samples)

        # Perturb videos accordingly and run inference on them
        return data, self.perturb_and_predict(
            data, batch_size, progress_bar, save_videos_path,
            load_videos_path, dont_predict, blur_mode=blur_mode, 
            blur_radius=blur_radius, blur_sigma=blur_sigma)

    def get_data(self, num_samples=5):
        """ Occludes multiple regions for each perturbed video (approximately
        half of the regions).

        Args:
            num_samples (int, optional): number of desired iterations of the
                perturbation loop. The number of perturbed videos obtained is
                equal to num_features*num_samples.

        Returns:
            data (2d numpy array): with shape: [num_samples+2, num_features]. This
                array will have 0 where a region is occluded for a specific
                instance, and 1 where not. The first row corresponds to the
                original video without perturbations and the second one to the
                fully occluded video.
        """

        # Initialize data
        data = np.zeros(shape=(num_samples*self.n_features, self.n_features),
                        dtype=int)

        for i in range(num_samples*self.n_features):

            # Get current feature
            i_feature = i % self.n_features

            # At the start of iteration, stablish a random order to add
            # the regions
            if i_feature == 0:
                rand_order = list(range(self.n_features))
                self.random_state.shuffle(rand_order)

            # Add the region to all next perturbed videos of the iteration
            data[i:i+self.n_features-i_feature, rand_order[i_feature]] = 1

        # Add original video and fully occluded video at the beginning
        data = np.vstack([
            np.ones(self.n_features, dtype=int),
            np.zeros(self.n_features, dtype=int),
            data])
        return data

    def get_non_perturbed_samples(self, num_samples):
        """ Computes which of the rows in the returned data correspond to
        non-perturbed samples.

        Args:
            num_samples (int): number of desired samples when calling to
                get_data.

        Returns:
            non_perturbed_samples (list): list with indexs of the non-perturbed 
                samples.
        """
        return [0, 1] + [1+i*self.n_features for i in range(1, num_samples+1)]


class ConvolutionalPerturber(Perturber):

    def __init__(self, video, classifier_fn, hide_color=None,
                 random_state=None, kernel_size=[5, 5, 5],
                 stride=[2, 2, 2], blur_mode='3d', blur_radius=25, blur_sigma=None):
        """ Perturber which occludes only one region for each perturbed video.
        The number of samples will be equal to the number of regions in the
        segmentation.

        Useful for computing the difference between occluding or not
        occluding each of the regions (ablation).

        The original video without occlusions is appended to the start.

        Args:
            video (4d numpy array): video to perturb, of shape:  [n_frames,
                height, width, 3].
            segments (3d numpy array): segmentation of the video, of shape:
                [n_frames, height, width], where each element indicates the
                region it belongs to.
            classifier_fn (_type_): function to predict the class of a video.
            hide_color (int | string, optional): color to use to occlude a region. 
                If None, the mean color of a region thrughout the video is used for 
                that region. If 'blur', the video is blurred using a Gaussian blur
                and used to perturb the input video. Defaults to None.
            random_state (int, optional): seed to use for replicating
                perturbations. Defaults to None.
            kernel_size (3d numpy array, optional): size of the kernel to use
                in each dimension. Each value represents the fraction of the
                size in each dimension, e.g., a value of 5 would make the
                kernel be a fifth of the total size in that dimension.
            stride (3d numpy array, optional): stride to use for the kernel in each
                dimension, as the fraction of the size of the kernel in that
                dimension. 0 represents a stride of one pixel (or frame)
                and 1 represents the size of the kernel.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be '3d' or '2d'. Defaults to '3d'.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.
        """
        super().__init__(video, None, classifier_fn, hide_color,
                         random_state, blur_mode=blur_mode, 
                         blur_radius=blur_radius, blur_sigma=blur_sigma),
        self.kernel_size = kernel_size
        self.stride = stride

    def perturb(self, batch_size=10, progress_bar=True, save_videos_path=None,
                load_videos_path=None, dont_predict=False, blur_mode=None, 
                blur_radius=25, blur_sigma=None):
        """ Generates videos with a single perturbation and their predictions.

        Args:
            batch_size (int, optional): number of videos to be passed together
                to the prediction function. Defaults to 10.
            progress_bar (bool, optional): Whether to display a progress bar or
                not. Defaults to True.
            save_videos_path (string, optional): path of a folder where
                perturbed instances should be stored. No videos will be stored
                if None. Defaults to None.
            load_videos_path (string, optional): path of the folder from where
                to load saved perturbed instances. Defaults to None.
            dont_predict (bool, optional): whether to call classifier_fn or
                not. Useful if only perturbations are wanted, or want to be
                stored. Defaults to False.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be None (no blur applied), '3d' or '2d'. Defaults to None.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.

        Returns:
            labels (2d numpy array): with shape [num_samples, num_classes], 
                conaining the prediction of classifier_fn for each class 
                and perturbed sample.
        """
        # Create data according to perturbation type
        return None, self.perturb_and_predict(
            batch_size, progress_bar, save_videos_path,
            load_videos_path, dont_predict, blur_mode=blur_mode, 
            blur_radius=blur_radius, blur_sigma=blur_sigma)

    def perturb_and_predict(self, batch_size=10, progress_bar=True,
                            save_videos_path=None, load_videos_path=None,
                            dont_predict=False, blur_mode=None, 
                            blur_radius=25, blur_sigma=None):
        """ Generates perturbed videos and their predictions. It has been
        adapted from the public LIME repository.

        Args:
            data (2d numpy array): array of shape: [num_samples, num_features]
                with 0 where a region is occluded for a specific instance, and
                1 where not.
            batch_size (int, optional): number of videos to be passed together
                to the prediction function. Defaults to 10.
            progress_bar (bool, optional): Whether to display a progress bar or
                not. Defaults to True.
            save_videos_path (string, optional): path of a folder where
                perturbed instances should be stored. No videos will be stored
                if None. Defaults to None.
            load_videos_path (string, optional): path of the folder from where
                to load saved perturbed instances. Defaults to None.
            dont_predict (bool, optional): whether to call classifier_fn or
                not. Useful if only perturbations are wanted, or want to be
                stored. Defaults to False.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be None (no blur applied), '3d' or '2d'. Defaults to None.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.

        Returns:
            labels (2d numpy array): with shape [num_samples, num_classes], 
                conaining the prediction of classifier_fn for each class 
                and perturbed sample.
        """

        kernel_size = self.kernel_size
        stride = self.stride
        labels = []
        vids = []

        # Progress bar
        if progress_bar:
            progress = \
                tqdm(range(int(np.prod([i*j-j+1 for i, j in
                                        zip(kernel_size, stride)])+1)))
        counter = 0

        # Unmodified video
        vids.append(self.video)

        for t in range(int(kernel_size[0]*stride[0]-stride[0]+1)):
            for y in range(int(kernel_size[1]*stride[1]-stride[1]+1)):
                for x in range(int(kernel_size[2]*stride[2]-stride[2]+1)):

                    # Generate perturbed video
                    if load_videos_path is None:

                        # Initialize perturbed video as a simple copy
                        temp = copy.deepcopy(self.video)

                        s1 = [round(a*b/(c*d)) for a, b, c, d in
                              zip([t, y, x], self.video.shape,
                                  stride, kernel_size)]
                        s2 = [round(a+(b/c)) for a, b, c in
                              zip(s1, self.video.shape, kernel_size)]
                        
                        if blur_mode is None:
                            temp[s1[0]:s2[0], s1[1]:s2[1], s1[2]:s2[2], :] = \
                                self.fudged_video[s1[0]:s2[0], s1[1]:s2[1],
                                                s1[2]:s2[2], :]
                        else:
                            mask = np.zeros(self.video.shape[:3], dtype='float32')
                            mask[s1[0]:s2[0], s1[1]:s2[1], s1[2]:s2[2]] = 1
                            mask = blur_video(mask, mode=blur_mode, 
                                              radius=blur_radius, sigma=blur_sigma)
                            temp = (temp * (1 - mask[...,None]) + self.fudged_video * 
                                    mask[...,None]).astype('uint8')

                        # Save perturbed video if specified
                        if save_videos_path is not None:
                            save_video(temp, save_videos_path,
                                       str(counter)+'.m4v')

                    # Load perturbed video from memory
                    else:
                        temp = load_video(os.path.join(load_videos_path,
                                                       str(counter)+'.m4v'))

                    # Add to the list of videos to predict if not dont_predict
                    if not dont_predict:
                        vids.append(temp)

                    # Predict all videos in list when the batch_size is full.
                    # Append results to list
                    if len(vids) == batch_size and not dont_predict:
                        preds = self.classifier_fn(np.array(vids))
                        labels.extend(preds)
                        vids = []
                        progress.update(batch_size)
                    elif dont_predict:
                        progress.update(1)

                    counter += 1

        # Predict remaining videos in list. Append results to list
        if len(vids) > 0 and not dont_predict:
            preds = self.classifier_fn(np.array(vids))
            labels.extend(preds)

            # Update progress
            progress.update(len(vids))

        return np.array(labels)

    def get_data(self):
        pass

    def get_non_perturbed_samples(self, num_samples=None):
        """ Computes which of the rows in the returned data correspond to
        non-perturbed samples.

        Args:
            num_samples (int): number of desired samples when calling to
                get_data.

        Returns:
            non_perturbed_samples (list): list with indexs of the non-perturbed 
                samples.
        """
        return [0]


class ShapPerturber(Perturber):

    def __init__(self, video, segments, classifier_fn, hide_color=None,
                 random_state=None, blur_mode='3d', blur_radius=25, blur_sigma=None):
        """ Perturber which occludes multiple regions for each perturbed video
        (approximately half of the regions). The number of perturbed videos
        can be chosen with num_samples.

        The original video without occlusions is appended to the start.

        Args:
            video (4d numpy array): video to perturb, of shape:  [n_frames,
                height, width, 3].
            segments (3d numpy array): segmentation of the video, of shape:
                [n_frames, height, width], where each element indicates the
                region it belongs to.
            classifier_fn (_type_): function to predict the class of a video
            hide_color (int | string, optional): color to use to occlude a region. 
                If None, the mean color of a region thrughout the video is used for 
                that region. If 'blur', the video is blurred using a Gaussian blur
                and used to perturb the input video. Defaults to None.
            random_state (int, optional): seed to use for replicating
                perturbations. Defaults to None.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be '3d' or '2d'. Defaults to '3d'.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.
        """
        super().__init__(video, segments, classifier_fn, hide_color,
                         random_state, blur_mode=blur_mode, 
                         blur_radius=blur_radius, blur_sigma=blur_sigma)

    def perturb(self, num_samples=500, batch_size=10,
                progress_bar=True, save_videos_path=None,
                load_videos_path=None, dont_predict=False,
                algorithm='kernel', blur_mode=None, 
                blur_radius=25, blur_sigma=None):
        """ Generates videos with multiple perturbations and their predictions.

        Args:
            num_samples (int, optional): number of desired perturbed videos.
                Defaults to 500.
            exact (bool, optional): whether to perturb exactly (True) or
                approximately (False) a (1-p) proportion of the regions.
                Defaults to False.
            p (float, optional): probability for each region to be occluded.
                Defaults to 0.5.
            batch_size (int, optional): number of videos to be passed together
                to the prediction function. Defaults to 10.
            progress_bar (bool, optional): Whether to display a progress bar or
                not. Defaults to True.
            save_videos_path (string, optional): path of a folder where
                perturbed instances should be stored. No videos will be stored
                if None. Defaults to None.
            load_videos_path (string, optional): path of the folder from where
                to load saved perturbed instances. Defaults to None.
            dont_predict (bool, optional): whether to call classifier_fn or
                not. Useful if only perturbations are wanted, or want to be
                stored. Defaults to False.
            algorithm (string, optional): algorithm to use for computing the
                Shapley values. Can be 'kernel' or 'permutation'. Defaults to
                'kernel'.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be None (no blur applied), '3d' or '2d'. Defaults to None.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.

        Returns:
            labels (2d numpy array): with shape [num_samples, num_classes], 
                conaining the prediction of classifier_fn for each class 
                and perturbed sample.
        """
        if algorithm == 'kernel':
            explainer = shap.KernelExplainer(
                partial(self.perturb_and_predict,
                        batch_size=batch_size,
                        progress_bar=progress_bar,
                        save_videos_path=save_videos_path,
                        load_videos_path=load_videos_path,
                        dont_predict=dont_predict,
                        blur_mode=blur_mode,
                        blur_radius=blur_radius,
                        blur_sigma=blur_sigma),
                np.zeros((1, self.n_features)))
            self.shapley_values = explainer.shap_values(
                np.ones((1, self.n_features)), nsamples=num_samples)
        elif algorithm == 'permutation':
            explainer = shap.Explainer(
                partial(self.perturb_and_predict,
                        batch_size=batch_size,
                        progress_bar=progress_bar,
                        save_videos_path=save_videos_path,
                        load_videos_path=load_videos_path,
                        dont_predict=dont_predict,
                        blur_mode=blur_mode,
                        blur_radius=blur_radius,
                        blur_sigma=blur_sigma),
                np.zeros((1, self.n_features)),
                algorithm='permutation')
            self.shapley_values = explainer(
                np.ones((1, self.n_features)), max_evals=num_samples)

        return self.data, self.labels

    def explain(self, label_to_explain=None,  algorithm='kernel'):

        # Label to explain
        if label_to_explain is None:
            label_to_explain = np.argmax(self.classifier_fn(
                np.array([self.video]))[0])

        if algorithm == 'kernel':
            return list(zip(range(self.n_features),
                            self.shapley_values[label_to_explain][0]))
        if algorithm == 'permutation':
            return list(zip(range(self.n_features),
                            self.shapley_values.values[0, :, label_to_explain]))
        # return self.shapley_values
