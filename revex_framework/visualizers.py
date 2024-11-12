import numpy as np
import cv2
from skimage.color import gray2rgb
from skimage.color import rgb2gray
from scipy import ndimage

from revex_framework.utils import blur_video, histogram_stretching, painter


class Visualizer(object):

    def __init__(self, video, segments, scores, gray=False):
        """ Base class for visualizing the explanations of a video.

        Args:
            video (4d numpy array): video being explained
            segments (3d numpy array): segmentation of the video, where each
                element indicates the region it belongs to.
            scores (list): a list of pairs [id_region, score], where id_region
                is in the range [0, n_regions] and the score is in the range
                [-1.0, 1.0] or [0.0, 1.0], dependingon the kind of
                explanations.
            gray (bool, optional): whether to convert video to gray scale for
                visualization. Useful to remove noise from background from the
                explanations. Defaults to False.
        """
        self.video = gray2rgb(rgb2gray(video)) if gray else video
        self.segments = segments
        self.scores = scores

    def get_score_map(self, th=None, top_k=None, min_accum=None,
                      pos_only=False, neg_only=False, improve_background=False,
                      weight_by_volume=False, blur_mode=None, blur_radius=25,
                      blur_sigma=None):
        """ Constructs an array of shape: [n_frames, height, width]
        representing the importance of each pixel of the video for the
        prediction confidence of a class.

        Args:
            th (float, optional): sets to 0 or to minimum value (depending on
                improve_background) all scores below it, useful for stablishing
                a minimum threshold and removing all regions below it. If None
                or min_accum not None, no regions are removed. Defaults to
                None.
            top_k (int, optional): finds the top_k most important regions and
                sets the other ones to 0 or to minimum value (depending on
                improve_background). Useful for visualizing only the most
                important regions, removing the rest. If None or min_accum not
                None, no regions are removed. Defaults to None.
            min_accum (float, optional): shows only most important regions
                summing up to min_accum % of the total importance, setting to
                0 or to minimum value (depending on improve_background) the
                rest. If None, no regions removed. Defaults to None.
            improve_background (bool, optional): whether to improve the
                contrast between regions when using optional attributes. If
                True: use minimum importance among all regions for removed
                regions. False: use 0 as importance for removed regions.
                Defaults to False.
            blur_mode (string, optional): mode to use for the blur filter. Can
                be None (no blur) '3d' or '2d'. Defaults to None.
            blur_radius (int, optional): radius of the blur filter. Defaults to
                25.
            blur_sigma (int, optional): sigma of the blur filter. If None, it
                is computed from the blur_radius. Defaults to None.

        Returns:
            score_map (3d numpy array): array of shape: [n_frames, height,
                width] representing the importance of each pixel of the video
                for the desired kind of explanations.
        """

        # Init score_map output array
        score_map = np.zeros(self.segments.shape, dtype=float)

        scores = self.scores

        # First of all, weight scores by the volume of its associated region
        # if requested
        if weight_by_volume:
            total_volume = np.prod(self.segments.shape)
            volume_regions = np.bincount(np.ravel(self.segments))
            scores = [[idx, score*volume/total_volume]
                      for volume, (idx, score) in zip(volume_regions, scores)]

        # Find minimum score among all, to use when improve_background=True
        min_score = np.min([np.abs(score) for _, score in scores])

        if pos_only:
            scores = [[idx, score] if score >= 0 else [idx, min_score if
                      improve_background else 0] for idx, score in scores]

        if neg_only:
            scores = [[idx, score] if score <= 0 else [idx, min_score if
                      improve_background else 0] for idx, score in scores]

        # Apply accumulated importance threshold
        if min_accum is not None:
            ids_scores = [(id_region, score, np.abs(score))
                          for id_region, score in scores]
            sorted_ids_scores = sorted(ids_scores, key=lambda tup: tup[2],
                                       reverse=True)
            score_sum = sum(abs_score for _, _, abs_score in sorted_ids_scores)
            accum = 0

            if improve_background:
                score_map += min_score

            for id_seg, score, abs_score in sorted_ids_scores:
                if accum/score_sum < min_accum:
                    score_map[self.segments == id_seg] = score
                    accum += abs_score
                else:
                    break

        # Don't apply accumulated importance threshold
        else:

            # Set importance of each region
            for id_seg, score in scores:
                score_map[self.segments == id_seg] = score

            # Apply minimum absolute threshold for a region
            if th is not None:
                score_map[np.abs(score_map) < th] = 0 \
                    if not improve_background else min_score

            # Leave only top_k important regions
            if top_k is not None:
                uniques = np.unique(np.abs(score_map))
                if uniques.size > top_k:
                    score_map[np.abs(score_map) < uniques[-top_k]] = 0 \
                        if not improve_background else min_score

        # Apply blur filter to the score_map
        if blur_mode is not None:
            score_map = blur_video(score_map, mode=blur_mode, 
                                   radius=blur_radius, sigma=blur_sigma)

        return score_map

    def visualize_on_video(self, rgb_score_map, alpha=0.5):
        """ Visualize a computed score_map with pixel importance over the
        explained video.

        Args:
            rgb_score_map (4d numpy array): array of shape: [n_frames, height,
                width, n_channels] representing the importance of each pixel of
                the video.
            alpha (float, optional): alpha value to use for the score_map
                (alpha value for the video is set to 1 - alpha), allowing to
                better visualize either the score_map or the video. Defaults to
                0.5.

        Returns:
            video (4d numpy array): the video resulting of the merge between
                the video being explained and the RGB explanations.
        """
        return painter(self.video, rgb_score_map, alpha)

    def visualize_on_video_clean(self, score_map, alpha=0.5, th=None, 
                                 min_accum=None):
        """ Visualize a computed score_map with pixel importance over the 
        explained video. The score_map is used to remove the heat map
        where the score is 0.

        Args:
            score_map (3d numpy array): array of shape: [n_frames, height,
                width] representing the importance of each pixel of
                the video.
            alpha (float, optional): alpha value to use for the score_map
                (alpha value for the video is set to 1 - alpha), allowing to
                better visualize either the score_map or the video. Defaults to
                0.5.
            th (optional): Threshold value for filtering the score_map. 
                Defaults to None.
            min_accum (optional): Minimum accumulation value for the score_map. 
                Defaults to None.

        Returns:
            video (4d numpy array): the video resulting of the merge between
                the video being explained and the RGB explanations.
        """

        # Resizing
        if self.video.shape[:3] != score_map.shape:
            scale = [self.video.shape[i] / score_map.shape[i]
                     for i in range(len(score_map.shape))]
            score_map = ndimage.zoom(score_map, scale, order=1)

        # 1st Histogram stretching
        score_map = histogram_stretching(score_map)

        # Apply minimum accumulated importance threshold
        if min_accum is not None:
            score_map2 = np.zeros_like(score_map)
            accum = 0
            score_sum = np.sum(score_map)
            for i in range(100, 0, -1):
                if accum/score_sum < min_accum:
                    mask = (score_map > (i-1)/100) & (score_map <= i/100)
                    score_map2[mask] = score_map[mask]
                    accum += np.sum(score_map[mask])
                else:
                    break
            score_map = score_map2
            score_map[score_map == 0] = np.min(score_map[score_map != 0])
            score_map = histogram_stretching(score_map)
        
        # Apply threshold
        if th is not None:
            mask = score_map <= th

        # [0, 1] to [0, 255]
        score_map = (score_map * 255).astype('uint8')

        rgb_score_map = np.zeros(self.video.shape, dtype='uint8')

        # Color map
        for i in range(score_map.shape[0]):
            rgb_score_map[i, ...] = cv2.applyColorMap(255 - score_map[i, ...], cv2.COLORMAP_JET)

        res = painter(self.video, rgb_score_map, alpha)

        # Useful to display only certain colors of the heatmap colormap (e.g. exclude blue)
        if th is not None:
            res[mask, :] = self.video[mask, :]

        return res

    def visualize(self):
        pass


class PosNegVisualizer(Visualizer):

    def __init__(self, video, segments, scores, gray=False):
        """ Allows LIME-like visualization of the explanations, using scores
        in the range [-1.0, 1.0]

        Args:
            video (4d numpy array): video being explained
            segments (3d numpy array): segmentation of the video, where each
                element indicates the region it belongs to.
            scores (list): a list of pairs [id_region, score], where id_region
                is in the range [0, n_regions] and the score is in the range
                [-1.0, 1.0].
            gray (bool, optional): whether to convert video to gray scale for
                visualization. Useful to remove noise from background from the
                explanations. Defaults to False.
        """
        super().__init__(video, segments, scores, gray)

    def visualize(self, score_map, hist_stretch=True, pos_channel=1,
                  neg_channel=0, pos_only=False, neg_only=False,
                  max_always=False):
        """ Computes an RGB video representation of the importance of the
        different regions for a target prediction. The importance can be
        positive or negative, depending on the contribution to the target
        class.

        Positive channel is set to 255 and the negative channel to 0 when a
        region contributes positively, and viceversa otherwise.

        The third channel is used to visualize the magnitude of the
        importance: white represents minimum importance (0) and black the
        maximum (255). This way, the region will be whiter when the region
        is less important, and redder, greener or bluer (depending on
        selected channel) when the region is more important (whether it is
        positive or negative importance).

        Args:
            score_map (3d numpy array): per-pixel importance array of shape:
                [n_frames, height, width].
            hist_stretch (bool, optional): whether to perform histogram
                strethcing for better visualizing the difference of importance
                between regions. Defaults to True.
            pos_channel (int, optional): Channel to set to 255 for regions with
                positive importance. Defaults to 1 (green).
            neg_channel (int, optional): Channel to set to 255 for regions with
                negative importance. . Defaults to 0 (red).
            pos_only (bool, optional): whether to show only regions with
                positive importance (True) or not (False). Defaults to False.
            neg_only (bool, optional): whether to show only regions with
                negative importance (True) or not (False). Defaults to False.
            max_always (bool, optional): whether to set importance of all non
                zero regions to maximum or not. Only useful when used on a
                score_map computed to show only some regions. Defaults to
                False.

        Returns:
            rgb_score_map (4d numpy array): an RGB video representation of the
                importance of each pixel for the target prediction.
        """

        # Resizing
        if self.video.shape[:3] != score_map.shape:
            scale = [self.video.shape[i] / score_map.shape[i]
                     for i in range(len(score_map.shape))]
            score_map = ndimage.zoom(score_map, scale, order=1)

        # Get only magnitudes, without sign for proper hist_stretch
        score_map_abs = np.abs(score_map)

        # Histogram stretching
        if hist_stretch:
            score_map_abs = histogram_stretching(score_map_abs)

        # Invert colors: less important is brighter
        score_map_abs = 1 - score_map_abs

        # Preserve or not the magnitude of the explanations
        if max_always:
            rgb_score_map = np.ones_like(self.video)
            rgb_score_map[score_map > 0, :] = 0
        else:
            rgb_score_map = gray2rgb(score_map_abs)

        if not pos_only:
            rgb_score_map[score_map < 0, neg_channel] = 1
        if not neg_only:
            rgb_score_map[score_map > 0, pos_channel] = 1

        # From float [0, 1] to uint8 [0, 255]
        return (rgb_score_map * 255).astype('uint8')


class HeatmapVisualizer(Visualizer):

    def __init__(self, video, segments, scores, gray=False):
        """ Allows RISE-like visualization of the explanations, using scores
        in the range [0.0, 1.0].

        Args:
            video (4d numpy array): video being explained
            segments (3d numpy array): segmentation of the video, where each
                element indicates the region it belongs to.
            scores (list): a list of pairs [id_region, score], where id_region
                is in the range [0, n_regions] and the score is in the range
                [0.0, 1.0].
            gray (bool, optional): whether to convert video to gray scale for
                visualization. Useful to remove noise from background from the
                explanations. Defaults to False.
        """
        super().__init__(video, segments, scores, gray)

    def visualize(self, score_map, hist_stretch=True,
                  colormap=cv2.COLORMAP_JET, invert_colormap=True,
                  neg_only=False, improve_background=False):
        """ Computes an RGB video representation of the importance of the
        different regions for a target prediction. The importance of a
        region can only be positive, since it represents the confidence
        value of the predictions when this region is not occluded.

        Since importance is in the range [0.0, 1.0], a colormap is used
        to visualize and contrast the importance of the different regions.

        Args:
            score_map (3d numpy array): per-pixel importance array of shape:
                [n_frames, height, width].
            hist_stretch (bool, optional): whether to perform histogram
                strethcing for better visualizing the difference of importance
                between regions. Defaults to True.
            colormap (one of cv2.ColormapTypes, optional): colormap to use
                to contrast importance between regions. Defaults to
                cv2.COLORMAP_JET, which is the one used in RISE explanations.
            invert_colormap (bool, optional): whether to invert the colormap
                for visualization. Defaults to True.
            neg_only (bool, optional): whether to show only regions with
                negative importance (True) or not (False). Defaults to False.
            improve_background (bool, optional): whether to improve the
                contrast between regions when using optional attributes. If
                True: use minimum importance among all regions for removed
                regions. False: use 0 as importance for removed regions.
                Defaults to False.

        Returns:
            rgb_score_map (4d numpy array): an RGB video representation of the
                importance of each pixel for the target prediction.
        """

        # Remove negative (or positive) importance
        score_map = score_map.copy()
        if not neg_only:
            if improve_background and score_map[score_map > 0].size > 0:
                score_map[score_map < 0] = np.min(score_map[score_map >= 0])
            else:
                score_map[score_map < 0] = 0
        else:
            if improve_background and score_map[score_map < 0].size > 0:
                score_map[score_map > 0] = np.max(score_map[score_map <= 0])
            else:
                score_map[score_map > 0] = 0
            score_map = -score_map

        # Histogram stretching
        if hist_stretch:
            score_map = histogram_stretching(score_map)

        # Invert colors: 0->blue, 255->red for default colormap
        if invert_colormap:
            score_map = 1 - score_map

        # Resizing
        if self.video.shape[:3] != score_map.shape:
            scale = [self.video.shape[i] / score_map.shape[i]
                     for i in range(len(score_map.shape))]
            score_map = ndimage.zoom(score_map, scale, order=1)

        # [0, 1] to [0, 255]
        score_map = (score_map * 255).astype('uint8')

        # Do not apply a colormap
        if colormap is None:
            return score_map
        # Apply the colormap to each frame
        else:
            rgb_score_map = np.zeros(self.video.shape, dtype='uint8')

            # Color map
            for i in range(score_map.shape[0]):
                rgb_score_map[i, ...] = cv2.applyColorMap(score_map[i, ...],
                                                          colormap)

        return rgb_score_map


class ConvolutionalVisualizer(HeatmapVisualizer):

    def __init__(self, video, scores, gray=False,
                 kernel_size=[5, 5, 5], stride=[2, 2, 2]):
        """ Allows RISE-like visualization of the explanations, using scores
        in the range [0.0, 1.0].

        Args:
            video (4d numpy array): video being explained
            scores (list): a list of pairs [id_region, score], where id_region
                is in the range [0, n_regions] and the score is in the range
                [0.0, 1.0].
            gray (bool, optional): whether to convert video to gray scale for
                visualization. Useful to remove noise from background from the
                explanations. Defaults to False.
            kernel_size (3d numpy array, optional): size of the kernel to use
                in each dimension. Each value represents the fraction of the
                size in each dimension, e.g., a value of 5 would make the
                kernel be a fifth of the total size in that dimension.
            stride (3d numpy array, optional): stride to use for the kernel in each
                dimension, as the fraction of the size of the kernel in that
                dimension. 0 represents a stride of one pixel (or frame)
                and 1 represents the size of the kernel.

        """
        super().__init__(video, None, scores, gray)
        self.kernel_size = kernel_size
        self.stride = stride

    def get_score_map(self, th=None, top_k=None, min_accum=None,
                      improve_background=False):
        """ Constructs an array of shape: [n_frames, height, width]
        representing the importance of each pixel of the video for the
        prediction confidence of a class.

        Args:
            th (float, optional): sets to 0 or to minimum value (depending on
                improve_background) all scores below it, useful for stablishing
                a minimum threshold and removing all regions below it. If None
                or min_accum not None, no regions are removed. Defaults to
                None.
            top_k (int, optional): finds the top_k most important regions and
                sets the other ones to 0 or to minimum value (depending on
                improve_background). Useful for visualizing only the most
                important regions, removing the rest. If None or min_accum not
                None, no regions are removed. Defaults to None.
            min_accum (float, optional): shows only most important regions
                summing up to min_accum % of the total importance, setting to
                0 or to minimum value (depending on improve_background) the
                rest. If None, no regions removed. Defaults to None.
            improve_background (bool, optional): whether to improve the
                contrast between regions when using optional attributes. If
                True: use minimum importance among all regions for removed
                regions. False: use 0 as importance for removed regions.
                Defaults to False.

        Returns:
            score_map (3d numpy array): array of shape: [n_frames, height,
                width] representing the importance of each pixel of the video
                for the desired kind of explanations.
        """
        kernel_size = self.kernel_size
        stride = self.stride

        # Init score_map output array
        score_map = np.zeros(
            shape=[int(i*j-j+1) for i, j in zip(kernel_size, stride)],
            dtype=float)

        # The higher the score the more important the region
        scores = [[idx, 1 - score] for idx, score in self.scores]

        # Find minimum score among all, to use when improve_background=True
        min_score = np.min([score for _, score in scores])

        # Apply accumulated importance threshold
        if min_accum is not None:
            ids_scores = [(id_region, score)
                          for id_region, score in scores]
            sorted_ids_scores = sorted(ids_scores, key=lambda tup: tup[1],
                                       reverse=True)
            score_sum = sum(score for _, score in sorted_ids_scores)
            accum = 0

            if improve_background:
                score_map += min_score

            for id_seg, score in sorted_ids_scores:
                if accum/score_sum < min_accum:
                    score_map[np.unravel_index(
                        int(id_seg), shape=score_map.shape)] = score
                    accum += score
                else:
                    break
        # Don't apply accumulated importance threshold
        # Set importance of each region
        else:
            counter = 0
            for t in range(score_map.shape[0]):
                for y in range(score_map.shape[1]):
                    for x in range(score_map.shape[2]):
                        score_map[t, y, x] = scores[counter][1]
                        counter += 1

            # Apply minimum absolute threshold for a region
            if th is not None:
                score_map[np.abs(score_map) < th] = 0 \
                    if not improve_background else min_score

            # Leave only top_k important regions
            if top_k is not None:
                uniques = np.unique(np.abs(score_map))
                if uniques.size > top_k:
                    score_map[np.abs(score_map) < uniques[-top_k]] = 0 \
                        if not improve_background else min_score

        return score_map
