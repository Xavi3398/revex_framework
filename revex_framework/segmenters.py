import os
import numpy as np
import cv2
from scipy import ndimage
import zipfile
from skimage.segmentation import slic
from tqdm.auto import tqdm

from revex_framework.utils import painter


class Segmenter(object):

    def __init__(self, video):
        """ Base class for segmenting a video into different non-overlapping
        regions (segments).

        Args:
            video (4d numpy array): video to segment, of shape: [n_frames,
                height, width, 3].
        """
        self.video = video

    def segment(self):
        pass

    def plot_segments(self, segments, kind='overlay', show_progress=False):
        """ Creates a video to visualize the segments on the video, in a
        similar way to 'label2rgb', from skimage.color.

        Args:
            segments (3d numpy array, optional): segments to visualize on the
                video.
            kind (str, optional): either 'overlay' (random color for each
                segment) or 'avg' (mean color for each segment). Defaults to
                'overlay'.
            show_progress (bool, optional): Whether to show a progress bar.
                Defaults to False.

        Returns:
            video (4d numpy array): video with the colored segments over it
        """
        return segments2colors(segments, self.video, kind, show_progress)


class GridSegmenter(Segmenter):

    def __init__(self, video):
        """ Segmenter for obtaining non-overlapping regions (segments) in the
        shape of a simple grid.

        Args:
            video (4d numpy array): video to segment, of shape: [n_frames,
                height, width, 3].
        """
        super().__init__(video)

    def segment(self, n_seg=[5, 10, 10]):
        """Segmentation in simple grid shape.

        Args:
            n_seg (list, optional): number of regions for the grid along each
                axis. Total of segments will be equal to the product of each
                element of the list. Defaults to [5, 10, 10] (500 segments).

        Returns:
            segments (3d numpy array): computed segments of shape [n_frames, height,
                width], where each element will indicate the region it
                belongs to.
        """
        segments = np.zeros(shape=self.video.shape[:3], dtype=int)
        slice_size = [self.video.shape[i] / n_seg[i]
                      for i in range(len(n_seg))]
        id_seg = 0
        for t in range(n_seg[0]):
            for y in range(n_seg[1]):
                for x in range(n_seg[2]):
                    segments[int(t*slice_size[0]):int((t+1)*slice_size[0]),
                             int(y*slice_size[1]):int((y+1)*slice_size[1]),
                             int(x*slice_size[2]):int((x+1)*slice_size[2])
                             ] = id_seg
                    id_seg += 1
        return segments


class RiseSegmenter(Segmenter):

    def __init__(self, video):
        """ Segmenter for obtaining non-overlapping regions (segments) in a
        RISE way: a small grid is constructed, which will be upscaled
        using linear interpolation in the perturbation phase.

        Args:
            video (4d numpy array): video to segment, of shape: [n_frames,
                height, width, 3].
        """
        super().__init__(video)

    def segment(self, n_seg=[5, 10, 10]):
        """ Segmentation in a RISE way: a small grid is constructed, which
        will be upscaled using linear interpolation in the perturbation
        phase.

        Args:
            n_seg (list, optional): number of regions for the grid along each
                axis. Total of segments will be equal to the product of each
                element of the list. Defaults to [5, 10, 10] (500 segments).

        Returns:
            segments(3d numpy array): computed segments of shape n_seg, 
                where each element will indicate the region it belongs to.
        """
        segments = np.zeros(shape=n_seg, dtype=int)
        id_seg = 0
        for t in range(n_seg[0]):
            for y in range(n_seg[1]):
                for x in range(n_seg[2]):
                    segments[t, y, x] = id_seg
                    id_seg += 1
        return segments


class SlicSegmenter(Segmenter):

    def __init__(self, video):
        """ Segmenter for obtaining non-overlapping regions (segments) using
        the SLIC technique.

        Args:
            video (4d numpy array): video to segment, of shape: [n_frames,
                height, width, 3].
        """
        super().__init__(video)

    def segment(self, n_segments=200, compactness=5, spacing=[0.2, 1, 1]):
        """ Segmentation for obtaining non-overlapping regions (segments)
        using the SLIC technique. The arguments are passed directly to
        the slic function, from skimage.segmentation.

        Args:
            n_segments (int, optional): The (approximate) number of labels in
                the segmented output video. Defaults to 200.
            compactness (int, optional): Balances color proximity and space
                proximity. Higher values give more weight to space proximity,
                making superpixel shapes more cubic. This parameter depends
                strongly on video contrast and on the shapes of objects in the
                video. We recommend exploring possible values on a log scale,
                e.g., 0.01, 0.1, 1, 10, 100, before refining around a chosen
                value. Defaults to 5.
            spacing (list, optional): The voxel spacing along each spatial
                dimension. By default, slic assumes uniform spacing (same
                voxel resolution along each spatial/temporal dimension). This
                parameter controls the weights of the distances along the
                spatial/temporal dimensions during k-means clustering.
                Defaults to [0.2,1,1], decreasing importance of temporal
                dimension.

        Returns:
            segments (3d numpy array): computed segments of shape [n_frames, 
                height, width], where each element will indicate the region it
                belongs to.
        """
        return slic(self.video, n_segments=n_segments, compactness=compactness,
                    spacing=spacing, start_label=0)


class OpticalFlowSegmenter(Segmenter):

    def __init__(self, video):
        """ Segmenter for obtaining non-overlapping regions (segments) using
        either SLIC or a grid to get segments in initial frame, and then
        optical flow to extend the segments to the remaining frames.

        Args:
            video (4d numpy array): video to segment, of shape: [n_frames,
                height, width, 3].
        """
        super().__init__(video)

    def segment(self, n_temp_slices=1, segments_initial_frames=None,
                seg_method='slic', of=None, of_method='farneback',
                n_segments=200, compactness=5, spacing=[1, 1],
                n_seg=[10, 10]):
        """ Segmentation for obtaining non-overlapping regions (segments)
        using either SLIC or a grid to get segments in initial frame,
        and then optical flow to extend the segments to the remaining
        frames.

        Args:
            n_temp_slices (int, optional): Number of temporal slices. If
                greater than 1, the resulting segments are equal to the
                concatenation of the segments computation of the video sliced
                into n_temp_slices. Defaults to 1.
            segments_initial_frames (list, optional): list where each element
                is the segmentation of the initial frame to use for each
                slice. If None, segmentation of these is done suing the method
                specified in seg_method. Defaults to None.
            seg_method (str, optional): method to use to segment the initial
                frame of each slice. Defaults to 'slic'.
            of (list, optional): list containing the computed optical flow for
                each frame. The optical flow should have the x flow on channel
                0 and the y flow on channel 1, in the same way as provided by
                function cv2.calcOpticalFlowFarneback. If None, Optical Flow
                will be computed according to of_method. Defaults to None.
            of_method (str, optional): method to use to compute Optical Flow.
                Currently only 'farneback' is available. Defaults to
                'farneback'.
            n_segments (int, optional): (approximate) number of segments to
                use when 'slic' is selected as seg_method. Defaults to 200.
            compactness (int, optional): compactness to use when 'slic' is
                selected as seg_method. Defaults to 5.
            spacing (list, optional): spacing to use when 'slic' is selected
                as seg_method. Note that in this case only spatial dimensions
                are considered. Defaults to [1,1].
            n_seg (list, optional): number of regions for the grid along each
                axis, when 'grid' is selected as seg_method. Note that in this
                case only spatial dimensions are considered. Defaults to
                [10,10].

        Returns:
            segments (3d numpy array): computed segments of shape [n_frames, 
                height, width], where each element will indicate the region it
                belongs to.
        """

        # Load segments if given
        if segments_initial_frames is None:
            segments_initial_frames = [None] * n_temp_slices

        # Size of each temporal slice
        frames_per_slice = self.video.shape[0] // n_temp_slices

        # First sliced video segments
        sliced_segments = self.__segment_slice(
            self.video[:frames_per_slice],
            segments_initial_frame=segments_initial_frames[0],
            seg_method=seg_method,
            of=of[:frames_per_slice] if of is not None else None,
            of_method=of_method,
            n_segments=n_segments,
            compactness=compactness,
            spacing=spacing,
            n_seg=n_seg)

        # Counter of segments
        total_segments = np.unique(sliced_segments).shape[0]

        for i in range(n_temp_slices-1):

            # Compute new slice's segments
            new_segments = self.__segment_slice(
                self.video[(i+1)*frames_per_slice:(i+2)*frames_per_slice],
                segments_initial_frame=segments_initial_frames[i],
                seg_method=seg_method,
                of=(of[(i+1)*frames_per_slice:(i+2)*frames_per_slice]
                    if of is not None else None),
                of_method=of_method,
                n_segments=n_segments,
                compactness=compactness,
                spacing=spacing,
                n_seg=n_seg)

            # Update counter of segments
            new_segments = new_segments + total_segments
            total_segments += np.unique(new_segments).shape[0]

            # Concatenate new segments
            sliced_segments = np.concatenate((sliced_segments, new_segments),
                                             axis=0)

        return sliced_segments

    def __segment_slice(self, video, segments_initial_frame=None,
                        seg_method='slic', of=None, of_method='farneback',
                        n_segments=200, compactness=5, spacing=[1, 1],
                        n_seg=[10, 10]):

        # Compute Optical Flow
        if of is None:
            flow_list = []
            for i in range(1, video.shape[0]):

                if of_method == 'farneback':
                    prev_frame = cv2.cvtColor(video[i-1], cv2.COLOR_BGR2GRAY)
                    next_frame = cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                else:
                    flow = of_method(video[i-1], video[i])

                flow_list.append(flow)

        else:
            flow_list = of

        # init optical flow on each axis
        segments_flow_y = np.zeros(shape=video.shape[:3], dtype='float32')
        segments_flow_x = np.zeros(shape=video.shape[:3], dtype='float32')

        # set first frame coordinates
        segments_flow_y[0], segments_flow_x[0] = \
            np.mgrid[0:video.shape[1], 0:video.shape[2]].astype('float32')

        # compute mapping coordinate of each frame pixel
        # new_coordinate = coordinate on last frame - computed movement
        # (optical flow)
        for i_flow in range(len(flow_list)):
            segments_flow_x[i_flow+1] = \
                segments_flow_x[i_flow] - flow_list[i_flow][..., 0]
            segments_flow_y[i_flow+1] = \
                segments_flow_y[i_flow] - flow_list[i_flow][..., 1]

        # Compute segments for first frame
        if segments_initial_frame is None:
            if seg_method == 'slic':
                segments_initial_frame = slic(
                    video[0, ...],
                    n_segments=n_segments,
                    compactness=compactness,
                    spacing=spacing,
                    start_label=0)
            elif seg_method == 'grid':
                segmenter = GridSegmenter(video)
                segments_initial_frame = segmenter.segment(n_seg=[1,]+n_seg)[0]

        # Apply optical flow remapping
        segments = np.zeros(shape=video.shape[:3],
                            dtype=segments_initial_frame.dtype)
        segments[0] = segments_initial_frame
        for i_frame in range(len(flow_list)):
            segments[i_frame+1] = cv2.remap(
                segments_initial_frame,
                segments_flow_x[i_frame],
                segments_flow_y[i_frame],
                cv2.INTER_NEAREST)

        return segments


def load_segmentation(path, compressed=True):
    """Loads segments from memory.

    Args:
        path (string): path to the segments file
        compressed (bool, optional): whether if the file is compressed
            (.zip) or not (.npy). Defaults to True.

    Returns:
        segments (3d numpy array): loaded segments
    """

    # Extract .npy file from .zip, load and delete the temporal .npy file
    if compressed:
        path_zip = path
        path_npy = path[:-3] + 'npy'
        with zipfile.ZipFile(path_zip, 'r') as file_zip:
            file_zip.extractall(os.path.dirname(path_npy))
        segments = np.load(path_npy)  # Load stored segmentation
        os.remove(path_npy)

    # Load the .npy file
    else:
        path_npy = path
        segments = np.load(path_npy)  # Load stored segmentation

    return segments


def save_segmentation(segments, path, compressed=True):
    """Saves segments to memory.

    Args:
        segments (3d numpy array): segments array to save
        path (string): path of the segments file
        compressed (bool, optional): whether if the file is compressed
            (.zip) or not (.npy). Defaults to True.
    """

    # Save first as .npy file, compress to .zip and remove temporal
    # .npy file
    if compressed:
        path_zip = path
        path_npy = path[:-3] + 'npy'
        np.save(path_npy, segments)
        with zipfile.ZipFile(path_zip, 'w') as file_zip:
            file_zip.write(path_npy, arcname=os.path.basename(path_npy),
                           compress_type=zipfile.ZIP_DEFLATED)
            os.remove(path_npy)

    # Save as .npy file
    else:
        path_npy = path
        np.save(path_npy, segments)


def segments2colors(segments, video, kind='overlay', show_progress=False):
    """ Shows the segmentation on the input video. Works in the same way as
    label2rgb from skimage.color, either using random or average colors to
    fill the different regions.

    Args:
        segments (3d numpy array): segmentation of the video, of shape:
            [n_frames, height, width], where each element represents the region
            (or segment) a pixel belongs to.
        video (4d numpy array): video that is being segmented.
        kind (str, optional): either 'overlay' to display a random color for
            each region or 'avg' to use the mean color of the region. When
            using 'overlay', the video is shown in the background, merging it
            with the segmentatino colors. Defaults to 'overlay'.
        show_progress (bool, optional): whether to show the progress of
            computing the colored video. Defaults to False.

    Returns:
        video (4d numpy array): video with the colored segments over it.
    """

    id_segments = np.unique(segments)
    colors = np.zeros(shape=segments.shape + (3,), dtype='uint8')
    progress = tqdm(id_segments) if show_progress else id_segments

    # if segments need rescale (RISE segmentation)
    if segments.shape != video.shape[:3]:
        scale = [video.shape[i] / segments.shape[i]
                 for i in range(len(segments.shape))]
        segments_big = ndimage.zoom(segments, scale, order=0)

    for id_seg in progress:
        mask = segments == id_seg

        if kind == 'overlay':
            colors[mask, :] = np.random.randint(0, 255, 3)
        elif kind == 'avg':
            if segments.shape != video.shape[:3]:
                mask_video = segments_big == id_seg
                colors[mask, :] = np.mean(video[mask_video, :], axis=0)
            else:
                colors[mask, :] = np.mean(video[mask, :], axis=0)

    # if segments need rescale (RISE segmentation)
    if segments.shape != video.shape[:3]:
        colors = ndimage.zoom(colors, scale + [1,], order=1)

    if kind == 'overlay':
        return painter(colors, video)
    else:
        return colors
