import numpy as np
import cv2
import os
import decord as de
from scipy import ndimage
from matplotlib import pyplot as plt


def blur_video(video, mode='2d', radius=2, sigma=None):
    """ Apply a GAussian blur filter to a video using OpenCV.

    Args:
        video (4d numpy array): video to blur.
        mode (string, optional): Mode to apply the blur filter. Should be
            either '2d' (apply 2d blur to every frame) or '3d'. Defaults to 
            '2d'.
        radius (int, optional): Radius of the blur filter (kernel_size = 
            2 * radius + 1). Defaults to 2.
            
        sigma (float, optional): Sigma of the blur filter. Defaults
            to None, in which case it is computed as 0.3 * ((kernel_size - 1)
            * 0.5 - 1) + 0.8.

    Returns:
        blurred_video (4d numpy array): video with the blur filter applied.
    """
    video_out = np.zeros_like(video)

    # Compute kernel size
    kernel_size = 2 * radius + 1

    # Compute sigma if not provided
    if sigma is None:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    # Apply 2d blur filter to each frame
    if mode == '2d':
        for i in range(video.shape[0]):
            video_out[i, ...] = cv2.GaussianBlur(video[i, ...], (kernel_size, kernel_size), sigma)
    
    # Apply 3d blur filter to each channel using ndimage
    elif mode == '3d':
        
        # 3 channels
        if len(video.shape) == 4:
            for i in range(video.shape[3]):
                video_out[..., i] = ndimage.gaussian_filter(video[..., i], sigma=sigma)
        # No channels
        else:
            video_out = ndimage.gaussian_filter(video, sigma=sigma)

    else:
        raise ValueError('mode should be either 2d or 3d')
    
    return video_out


def histogram_stretching(video, h_min=0, h_max=1):
    """ Apply histogram stretching to a video.

    Args:
        video (4d numpy array): video to stretch.
        h_min (float, optional): Minimum value of the stretched video.
            Defaults to 0.
        h_max (float, optional): Maximum value of the stretched video.
            Defaults to 1.

    Returns:
        stretched_video (4d numpy array): video with the histogram stretched.
    """
    max_value = np.max(video)
    min_value = np.min(video)
    if max_value > 0 and min_value != max_value:
        return h_min+(h_max-h_min)*(video-min_value)/(max_value-min_value)
    else:
        return video


def painter(vid1, vid2, alpha2=0.5, improve=False):
    """ Merges two videos into one, according to alpha factor.

    Args:
        vid1: 1st video
        vid2: 2nd video
        alpha2: Importance of 2nd video. 1 maximum, 0 minimum.

    Returns:
        merged_video (4d numpy array): video with the two videos merged.
    """
    result = (vid1.astype('float') * (1 - alpha2)
            + vid2.astype('float') * alpha2).astype('uint8')
            
    return result


def show_video(video, bgr=False):
    """ Shows a video in a window using OpenCV.

    Args:
        video (4d numpy array): video to show
        bgr (bool, optional): Whether the video is already in BGR format or
            not. Defaults to False.
    """
    for n_frame in range(video.shape[0]):
        frame = video[n_frame, ...] if bgr \
                else cv2.cvtColor(video[n_frame, ...], cv2.COLOR_RGB2BGR)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q') or k == 27:
            break
        cv2.imshow('frame', frame)
    cv2.destroyAllWindows()


def load_video(path, lib='decord'):
    """ Load a video from path.

    Args:
        path (string): path to the video.

    Returns:
        video (4d numpy array): loaded video as a numpy array of shape:
            [n_frames, height, width, channels]
    """
    if lib == 'cv2':
        video = []
        cap = cv2.VideoCapture(path)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                video.append(frame)
            else:
                break
        cap.release()
        video = np.array(video)
    else:
        vr = de.VideoReader(path)
        video = vr.get_batch(range(0, len(vr), 1)).asnumpy()
    return video


def save_video(video, path, video_name=None, fps=30, bgr=False):
    """ Save a video to a path.

    Args:
        video (4d numpy array): video to save
        path (string): path to the folder where the video should be stored
            if video_name is not None. Otherwise, the path should already
            contain the video_name.
        video_name (string, optional): name of the video to store. Defaults to
            None.
        fps (int, optional): Frames per second to use to store the video.
            Defaults to 30.
        bgr (bool, optional): Whether the video is already in BGR format or
            not. Defaults to False.
    """
    out = cv2.VideoWriter(
        os.path.join(path, video_name) if video_name is not None else path,
        cv2.VideoWriter_fourcc(*'DIVX'),
        fps, video.shape[1:3][::-1])
    for n_frame in range(video.shape[0]):
        frame = video[n_frame, ...] if bgr \
            else cv2.cvtColor(video[n_frame, ...], cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()


def resize_video(video, scale_x, scale_y, interpolation=cv2.INTER_LINEAR):
    """ Resize video to new height and width frame by frame using OpenCV.

    Args:
        video (4d numpy array): video to resize.
        scale_x (float): new video width.
        scale_y (float): new video height.
        interpolation (_type_, optional): Interpolation to use when resizing
            the frames. Should be one of cv2.InterpolationFlags. Defaults to
            cv2.INTER_LINEAR.

    Returns:
        resized_video (4d numpy array): video with the new size.
    """
    new_w = int(video.shape[2] * scale_x)
    new_h = int(video.shape[1] * scale_y)

    # Case of gray video
    if len(video.shape) == 3:
        video_out = np.zeros(shape=(video.shape[0], new_h, new_w),
                             dtype='uint8')

    # Case of RGB video
    else:
        video_out = np.zeros(shape=(video.shape[0], new_h, new_w, 3),
                             dtype='uint8')

    # Resize frames
    for i in range(video.shape[0]):
        frame = video[i, ...]
        video_out[i, ...] = cv2.resize(frame, dsize=(new_w, new_h),
                                       interpolation=interpolation)

    return video_out


def center_crop_video(video, size):
    """ Takes a squared center of size 'size' from the video (only on spatial
    dimensions).

    Args:
        video (4d numpy array): video to center crop.
        size (int): size of the center square to take.

    Returns:
        cropped_video (4d numpy array): video with the center crop, with shape
            [video.shape[0], size, size, video.shape[3]]
    """
    return video[
        :,
        video.shape[1]//2-size//2:video.shape[1]//2 + size//2,
        video.shape[2]//2-size//2:video.shape[2]//2 + size//2,
        ...]


def rgb2bgr(video):
    """ Change channel order of video to transform it from RGB to BGR.

    Args:
        video (4d numpy array): video where to perform change.

    Returns:
       bgr_video (4d numpy array): video with channel order modified.
    """
    return video[..., ::-1]


def bgr2rgb(video):
    """ Change channel order of video to transform it from BGR to RGB. The
        functionallity is the same than rgb2bgr.

    Args:
        video (4d numpy array): video where to perform change.

    Returns:
       rgb_video (4d numpy array): video with channel order modified.
    """
    return rgb2bgr(video)


def plot_frames(video, n_frames=3, figsize=3):
    """ Show frames of a video at equal steps in the same plot.

    Args:
        video (4d numpy array): video to plot.
        n_frames (int, optional): number of frames to plot. Defaults to 3.
    """

    fig, axs = plt.subplots(1, n_frames, figsize=(figsize*n_frames, figsize))
    for i, ax in enumerate(axs):
        ax.imshow(video[i*int((len(video) - 1)/(n_frames-1))])
        ax.axis('off')
    plt.show()