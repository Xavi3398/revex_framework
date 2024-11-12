from revex_framework.segmenters import *
from revex_framework.perturbers import *
from revex_framework.explainers import *
from revex_framework.visualizers import *

import cv2
from tqdm import tqdm


class Xai(object):

    # Default parameters for segmentation
    seg_dict_default = {
        'type': 'slic',
        'n_temp_slices': 1,
        'segments_initial_frames': None,
        'seg_method': 'slic',
        'of': None,
        'of_method': 'farneback',
        'n_seg': [5, 10, 10],
        'n_segments': 200,
        'compactness': 5,
        'spacing': [0.2, 1, 1],
    }

    # Default parameters for perturbations
    per_dict_default = {
        'type': 'multiple',
        'hide_color': None,
        'random_state': None,
        'num_samples': 500,
        'batch_size': 10,
        'progress_bar': True,
        'save_videos_path': None,
        'load_videos_path': None,
        'dont_predict': False,
        'exact': False,
        'p': 0.5,
        'kernel_size': [5, 5, 5],
        'stride': [2, 2, 2],
        'blur_init_mode':'3d', 
        'blur_init_radius': 51,
        'blur_init_sigma':None,
        'blur_mode':'3d', 
        'blur_radius': 51,
        'blur_sigma':None,
    }

    # Default parameters for explanations
    exp_dict_default = {
        'type': 'lime',
        'random_state': None,
        'label_to_explain': None,
        'kernel_width': .25,
        'kernel': None,
        'verbose': False,
        'feature_selection': 'auto',
        'num_features': 100000,
        'distance_metric': 'cosine',
        'model_regressor': None,
        'invert': False
    }

    # Default parameters for visualization
    vis_dict_default = {
        'type': 'posneg',
        'th': None,
        'top_k': None,
        'min_accum': None,
        'improve_background': False,
        'weight_by_volume': False,
        'hist_stretch': True,
        'pos_channel': 1,
        'neg_channel': 0,
        'pos_only': False,
        'neg_only': False,
        'colormap': cv2.COLORMAP_JET,
        'invert_colormap': True,
        'alpha': 0.5,
        'kernel_size': [5, 5, 5],
        'stride': [2, 2, 2],
        'blur_mode': None,
        'blur_radius': 25,
        'blur_sigma': None
    }

    def __init__(self, video, classifier_fn, pipeline):
        """ Class to define a full explanation pipeline, defined by the
        parameters used for each of the steps: segmentation, perturbation,
        explanation and visualization.

        Args:
            video (4d numpy array): video to explain, of shape: [n_frames,
                height, width, channels]
            seg_dict (dictionary): parameters to use for segmentation.
            per_dict (dictionary): parameters to use for perturbation.
            exp_dict (dictionary): parameters to use for explanation.
            vis_dict (dictionary): parameters to use for visualization.
        """

        self.update_dict(pipeline.seg, 'seg')
        self.update_dict(pipeline.per, 'per')
        self.update_dict(pipeline.exp, 'exp')
        self.update_dict(pipeline.vis, 'vis')

        self.classifier_fn = classifier_fn
        self.video = video

    def update_dict(self, task_dict, task):
        """ Updates used dictionary for a given task. First initializes 
        with default values and then updates with given dictionary.

        Args:
            task_dict (dictionary): dictionary to update.
            task (string): one of: 'seg', 'per', 'exp' or 'vis'.
        """
        if task == 'seg':
            self.seg_dict = Xai.seg_dict_default.copy()
            self.seg_dict.update(task_dict)
        elif task == 'per':
            self.per_dict = Xai.per_dict_default.copy()
            self.per_dict.update(task_dict)
        elif task == 'exp':
            self.exp_dict = Xai.exp_dict_default.copy()
            self.exp_dict.update(task_dict)
        elif task == 'vis':
            self.vis_dict = Xai.vis_dict_default.copy()
            self.vis_dict.update(task_dict)

    def segment(self):
        """ Segments the video using the configuration present in seg_dict.
        Modifies self.segmenter and self.segments.
        """

        if self.seg_dict['type'] == 'grid':
            self.segmenter = GridSegmenter(self.video)
            self.segments = self.segmenter.segment(
                n_seg=self.seg_dict['n_seg'])

        elif self.seg_dict['type'] == 'rise':
            self.segmenter = RiseSegmenter(self.video)
            self.segments = self.segmenter.segment(
                n_seg=self.seg_dict['n_seg'])

        elif self.seg_dict['type'] == 'slic':
            self.segmenter = SlicSegmenter(self.video)
            self.segments = self.segmenter.segment(
                n_segments=self.seg_dict['n_segments'],
                compactness=self.seg_dict['compactness'],
                spacing=self.seg_dict['spacing'],)

        elif self.seg_dict['type'] == 'flow':
            self.segmenter = OpticalFlowSegmenter(self.video)
            self.segments = self.segmenter.segment(
                n_temp_slices=self.seg_dict['n_temp_slices'],
                segments_initial_frames=self.seg_dict[
                    'segments_initial_frames'],
                seg_method=self.seg_dict['seg_method'],
                of=self.seg_dict['of'],
                of_method=self.seg_dict['of_method'],
                n_segments=self.seg_dict['n_segments'],
                compactness=self.seg_dict['compactness'],
                spacing=self.seg_dict['spacing'],
                n_seg=self.seg_dict['n_seg'])

        elif self.seg_dict['type'] == 'void':
            self.segmenter = None
            self.segments = None

    def perturb(self):
        """ Generates perturbations of the video and makes predictions of them
        using the configuration present in per_dict. Modifies self.perturber, 
        self.data, self.labels and self.non_perturbed_samples.
        """

        if self.per_dict['type'] == 'multiple':
            self.perturber = MultiplePerturber(
                video=self.video,
                segments=self.segments,
                classifier_fn=self.classifier_fn,
                hide_color=self.per_dict['hide_color'],
                random_state=self.per_dict['random_state'],
                blur_mode=self.per_dict['blur_init_mode'],
                blur_radius=self.per_dict['blur_init_radius'],
                blur_sigma=self.per_dict['blur_init_sigma'])
            self.data, self.labels = self.perturber.perturb(
                num_samples=self.per_dict['num_samples'],
                exact=self.per_dict['exact'],
                p=self.per_dict['p'],
                batch_size=self.per_dict['batch_size'],
                progress_bar=self.per_dict['progress_bar'],
                save_videos_path=self.per_dict['save_videos_path'],
                load_videos_path=self.per_dict['load_videos_path'],
                dont_predict=self.per_dict['dont_predict'],
                blur_mode=self.per_dict['blur_mode'],
                blur_radius=self.per_dict['blur_radius'],
                blur_sigma=self.per_dict['blur_sigma'])
            self.non_perturbed_samples = \
                self.perturber.get_non_perturbed_samples(
                    self.per_dict['num_samples'])

        elif self.per_dict['type'] == 'single':
            self.perturber = SinglePerturber(
                video=self.video,
                segments=self.segments,
                classifier_fn=self.classifier_fn,
                hide_color=self.per_dict['hide_color'],
                random_state=self.per_dict['random_state'],
                blur_mode=self.per_dict['blur_init_mode'],
                blur_radius=self.per_dict['blur_init_radius'],
                blur_sigma=self.per_dict['blur_init_sigma'])
            self.data, self.labels = self.perturber.perturb(
                batch_size=self.per_dict['batch_size'],
                progress_bar=self.per_dict['progress_bar'],
                save_videos_path=self.per_dict['save_videos_path'],
                load_videos_path=self.per_dict['load_videos_path'],
                dont_predict=self.per_dict['dont_predict'],
                blur_mode=self.per_dict['blur_mode'],
                blur_radius=self.per_dict['blur_radius'],
                blur_sigma=self.per_dict['blur_sigma'])
            self.non_perturbed_samples = \
                self.perturber.get_non_perturbed_samples(
                    self.per_dict['num_samples'])

        elif self.per_dict['type'] == 'convolutional':
            self.perturber = ConvolutionalPerturber(
                video=self.video,
                classifier_fn=self.classifier_fn,
                hide_color=self.per_dict['hide_color'],
                random_state=self.per_dict['random_state'],
                kernel_size=self.per_dict['kernel_size'],
                stride=self.per_dict['stride'],
                blur_mode=self.per_dict['blur_init_mode'],
                blur_radius=self.per_dict['blur_init_radius'],
                blur_sigma=self.per_dict['blur_init_sigma'])
            self.data, self.labels = self.perturber.perturb(
                batch_size=self.per_dict['batch_size'],
                progress_bar=self.per_dict['progress_bar'],
                save_videos_path=self.per_dict['save_videos_path'],
                load_videos_path=self.per_dict['load_videos_path'],
                dont_predict=self.per_dict['dont_predict'],
                blur_mode=self.per_dict['blur_mode'],
                blur_radius=self.per_dict['blur_radius'],
                blur_sigma=self.per_dict['blur_sigma'])
            self.non_perturbed_samples = \
                self.perturber.get_non_perturbed_samples(
                    self.per_dict['num_samples'])

        elif self.per_dict['type'] == 'allbutone':
            self.perturber = AllButOnePerturber(
                video=self.video,
                segments=self.segments,
                classifier_fn=self.classifier_fn,
                hide_color=self.per_dict['hide_color'],
                random_state=self.per_dict['random_state'],
                blur_mode=self.per_dict['blur_init_mode'],
                blur_radius=self.per_dict['blur_init_radius'],
                blur_sigma=self.per_dict['blur_init_sigma'])
            self.data, self.labels = self.perturber.perturb(
                batch_size=self.per_dict['batch_size'],
                progress_bar=self.per_dict['progress_bar'],
                save_videos_path=self.per_dict['save_videos_path'],
                load_videos_path=self.per_dict['load_videos_path'],
                dont_predict=self.per_dict['dont_predict'],
                blur_mode=self.per_dict['blur_mode'],
                blur_radius=self.per_dict['blur_radius'],
                blur_sigma=self.per_dict['blur_sigma'])
            self.non_perturbed_samples = \
                self.perturber.get_non_perturbed_samples(
                    self.per_dict['num_samples'])

        elif self.per_dict['type'] == 'accum':
            self.perturber = AccumPerturber(
                video=self.video,
                segments=self.segments,
                classifier_fn=self.classifier_fn,
                hide_color=self.per_dict['hide_color'],
                random_state=self.per_dict['random_state'],
                blur_mode=self.per_dict['blur_init_mode'],
                blur_radius=self.per_dict['blur_init_radius'],
                blur_sigma=self.per_dict['blur_init_sigma'])
            self.data, self.labels = self.perturber.perturb(
                num_samples=self.per_dict['num_samples'],
                batch_size=self.per_dict['batch_size'],
                progress_bar=self.per_dict['progress_bar'],
                save_videos_path=self.per_dict['save_videos_path'],
                load_videos_path=self.per_dict['load_videos_path'],
                dont_predict=self.per_dict['dont_predict'],
                blur_mode=self.per_dict['blur_mode'],
                blur_radius=self.per_dict['blur_radius'],
                blur_sigma=self.per_dict['blur_sigma'])
            self.non_perturbed_samples = \
                self.perturber.get_non_perturbed_samples(
                    self.per_dict['num_samples'])

        elif self.per_dict['type'] == 'shap':
            self.perturber = ShapPerturber(
                video=self.video,
                segments=self.segments,
                classifier_fn=self.classifier_fn,
                hide_color=self.per_dict['hide_color'],
                random_state=self.per_dict['random_state'],
                blur_mode=self.per_dict['blur_init_mode'],
                blur_radius=self.per_dict['blur_init_radius'],
                blur_sigma=self.per_dict['blur_init_sigma'])
            self.data, self.labels = self.perturber.perturb(
                num_samples=self.per_dict['num_samples'],
                batch_size=self.per_dict['batch_size'],
                progress_bar=self.per_dict['progress_bar'],
                save_videos_path=self.per_dict['save_videos_path'],
                load_videos_path=self.per_dict['load_videos_path'],
                dont_predict=self.per_dict['dont_predict'],
                algorithm=self.per_dict['algorithm'],
                blur_mode=self.per_dict['blur_mode'],
                blur_radius=self.per_dict['blur_radius'],
                blur_sigma=self.per_dict['blur_sigma'])
            self.non_perturbed_samples = None

    def explain(self):
        """ Explains the video using the configuration present in exp_dict.
        Modifies self.explainer and self.scores.
        """

        if self.exp_dict['type'] == 'lime':
            self.explainer = LimeExplainer(
                data=self.data,
                labels=self.labels)
            self.scores = self.explainer.explain(
                label_to_explain=self.exp_dict['label_to_explain'],
                kernel_width=self.exp_dict['kernel_width'],
                kernel=self.exp_dict['kernel'],
                verbose=self.exp_dict['verbose'],
                feature_selection=self.exp_dict['feature_selection'],
                num_features=self.exp_dict['num_features'],
                distance_metric=self.exp_dict['distance_metric'],
                model_regressor=self.exp_dict['model_regressor'],
                random_state=self.exp_dict['random_state'],
                non_perturbed_samples=self.non_perturbed_samples)

        elif self.exp_dict['type'] == 'mean':
            self.explainer = MeanExplainer(
                data=self.data,
                labels=self.labels)
            self.scores = self.explainer.explain(
                label_to_explain=self.exp_dict['label_to_explain'])

        elif self.exp_dict['type'] == 'difference':
            self.explainer = DifferenceExplainer(
                data=self.data,
                labels=self.labels)
            self.scores = self.explainer.explain(
                label_to_explain=self.exp_dict['label_to_explain'],
                invert=self.exp_dict['invert'])

        elif self.exp_dict['type'] == 'value':
            self.explainer = ValueExplainer(
                data=self.data,
                labels=self.labels)
            self.scores = self.explainer.explain(
                label_to_explain=self.exp_dict['label_to_explain'])

        elif self.exp_dict['type'] == 'shap':
            self.explainer = ShapExplainer(
                data=self.data,
                labels=self.labels)
            self.scores = self.explainer.explain(
                label_to_explain=self.exp_dict['label_to_explain'])

        elif self.exp_dict['type'] == 'from_perturber':
            self.scores = self.perturber.explain(
                label_to_explain=self.exp_dict['label_to_explain'],
                algorithm=self.per_dict['algorithm'])

    def visualize(self):
        """ Visualizes the explanations of a video using the configuration
        present in vis_dict. Modifies self.visualizer, self.score_map, 
        self.rgb_score_map and self.exp_vid.
        """

        if self.vis_dict['type'] == 'posneg':
            self.visualizer = PosNegVisualizer(
                video=self.video,
                segments=self.segments,
                scores=self.scores)
            self.score_map = self.visualizer.get_score_map(
                th=self.vis_dict['th'],
                top_k=self.vis_dict['top_k'],
                min_accum=self.vis_dict['min_accum'],
                improve_background=self.vis_dict['improve_background'],
                pos_only=self.vis_dict['pos_only'],
                neg_only=self.vis_dict['neg_only'],
                weight_by_volume=self.vis_dict['weight_by_volume'],
                blur_mode=self.vis_dict['blur_mode'],
                blur_radius=self.vis_dict['blur_radius'],
                blur_sigma=self.vis_dict['blur_sigma'])
            self.rgb_score_map = self.visualizer.visualize(
                score_map=self.score_map,
                hist_stretch=self.vis_dict['hist_stretch'],
                pos_channel=self.vis_dict['pos_channel'],
                neg_channel=self.vis_dict['neg_channel'],
                pos_only=self.vis_dict['pos_only'],
                neg_only=self.vis_dict['neg_only'])
            self.exp_vid = self.visualizer.visualize_on_video(
                rgb_score_map=self.rgb_score_map,
                alpha=self.vis_dict['alpha'])

        elif self.vis_dict['type'] == 'heatmap':
            self.visualizer = HeatmapVisualizer(
                video=self.video,
                segments=self.segments,
                scores=self.scores)
            self.score_map = self.visualizer.get_score_map(
                th=self.vis_dict['th'],
                top_k=self.vis_dict['top_k'],
                min_accum=self.vis_dict['min_accum'],
                pos_only=self.vis_dict['pos_only'],
                neg_only=self.vis_dict['neg_only'],
                improve_background=self.vis_dict['improve_background'],
                weight_by_volume=self.vis_dict['weight_by_volume'],
                blur_mode=self.vis_dict['blur_mode'],
                blur_radius=self.vis_dict['blur_radius'],
                blur_sigma=self.vis_dict['blur_sigma'])
            self.rgb_score_map = self.visualizer.visualize(
                score_map=self.score_map,
                hist_stretch=self.vis_dict['hist_stretch'],
                colormap=self.vis_dict['colormap'],
                invert_colormap=self.vis_dict['invert_colormap'],
                neg_only=self.vis_dict['neg_only'],
                improve_background=self.vis_dict['improve_background'])
            self.exp_vid = self.visualizer.visualize_on_video(
                rgb_score_map=self.rgb_score_map,
                alpha=self.vis_dict['alpha']
            )

        elif self.vis_dict['type'] == 'convolutional':
            self.visualizer = ConvolutionalVisualizer(
                video=self.video,
                scores=self.scores,
                kernel_size=self.vis_dict['kernel_size'],
                stride=self.vis_dict['stride'])
            self.score_map = self.visualizer.get_score_map(
                th=self.vis_dict['th'],
                top_k=self.vis_dict['top_k'],
                min_accum=self.vis_dict['min_accum'],
                improve_background=self.vis_dict['improve_background'])
            self.rgb_score_map = self.visualizer.visualize(
                score_map=self.score_map,
                hist_stretch=self.vis_dict['hist_stretch'],
                colormap=self.vis_dict['colormap'],
                invert_colormap=self.vis_dict['invert_colormap'],
                neg_only=self.vis_dict['neg_only'],
                improve_background=self.vis_dict['improve_background'])
            self.exp_vid = self.visualizer.visualize_on_video(
                rgb_score_map=self.rgb_score_map,
                alpha=self.vis_dict['alpha']
            )

    def run_pipeline(self, verbose=False):
        """ Run the whole explanation process.

        Args:
            verbose (bool, optional): whether to display messages about the
                current state of the explanation execution. Defaults to False.
        """

        # Segmentation
        if verbose:
            print('segmenting, type:', self.seg_dict['type'])
        self.segment()

        # Perturbation
        if verbose:
            print('perturbing, type:', self.per_dict['type'])
        self.perturb()

        # Explanation
        if verbose:
            print('explaining, type:', self.exp_dict['type'])
        self.explain()

        # Visualization
        if verbose:
            print('visualizing, type:', self.vis_dict['type'])
        self.visualize()


class XaiMerge(object):

    # Default parameters for visualization of merged explanations
    vis_merged_dict_default = {
        'type': 'lime',
        'th': None,  # Not used
        'top_k': None,  # Not used
        'min_accum': None,  # Not used
        'improve_background': False,  # Not used
        'hist_stretch': True,
        'pos_channel': 1,
        'neg_channel': 0,
        'pos_only': False,
        'neg_only': False,
        'colormap': cv2.COLORMAP_JET,
        'invert_colormap': True,
        'alpha': 0.5,
        'alpha_list': None,
        'mean_type': 'harmonic'
    }

    def __init__(self, video, classifier_fn, pipelines, vis_merged_dict,
                 show_progress=True):
        """ Define different explanation pipelines on the same video
        and merge the results into one.

        Args:
            video (4d numpy array): video to explain, of shape: [n_frames,
                height, width, channels].
            classifier_fn (function): function to predict the class of a video.
            pipelines (list): list of Pipeline objects.
            vis_merged_dict (dictionary): parameters to use for visualization
                of the merged explanations.
            show_progress (bool, optional): whether to show progress when
                computing the different parts of the pipeline for each Xai
                object or not. Defaults to True.
        """

        self.video = video
        self.classifier_fn = classifier_fn
        self.show_progress = show_progress

        # Initialize with default values and update parameters
        self.vis_merged_dict = XaiMerge.vis_merged_dict_default
        self.vis_merged_dict.update(vis_merged_dict)
        self.xais = [Xai(video, classifier_fn, p) for p in pipelines]

    def segment(self):
        """ Segment for all Xai objects.
        """
        progress = tqdm(self.xais) if self.show_progress else self.xais
        for xai in progress:
            xai.segment()

    def perturb(self):
        """ Perturb for all Xai objects.
        """
        progress = tqdm(self.xais) if self.show_progress else self.xais
        for xai in progress:
            xai.perturb()

    def explain(self):
        """ Explain for all Xai objects.
        """
        progress = tqdm(self.xais) if self.show_progress else self.xais
        for xai in progress:
            xai.explain()

    def visualize(self):
        """ Merges different explanations from different Xai objects into one
        and visualizes them on one sole video.
        """

        # Compute the merged score_map
        self.score_map = self.get_score_map(
            alpha_list=self.vis_merged_dict['alpha_list'],
            mean_type=self.vis_merged_dict['mean_type'])

        if self.vis_merged_dict['type'] == 'posneg':
            self.visualizer = PosNegVisualizer(
                video=self.video,
                segments=None,
                scores=None)
            self.rgb_score_map = self.visualizer.visualize(
                score_map=self.score_map,
                hist_stretch=self.vis_merged_dict['hist_stretch'],
                pos_channel=self.vis_merged_dict['pos_channel'],
                neg_channel=self.vis_merged_dict['neg_channel'],
                pos_only=self.vis_merged_dict['pos_only'],
                neg_only=self.vis_merged_dict['neg_only'])

        elif self.vis_merged_dict['type'] == 'heatmap':
            self.visualizer = HeatmapVisualizer(
                video=self.video,
                segments=None,
                scores=None)
            self.rgb_score_map = self.visualizer.visualize(
                score_map=self.score_map,
                hist_stretch=self.vis_merged_dict['hist_stretch'],
                colormap=self.vis_merged_dict['colormap'],
                invert_colormap=self.vis_merged_dict['invert_colormap'],
                neg_only=self.vis_merged_dict['neg_only'],
                improve_background=self.vis_merged_dict['improve_background'])

        elif self.vis_merged_dict['type'] == 'convolutional':
            raise ValueError("Convolutional visualization cannot be used \
                             for merged visualization.")

        # Visualize results on original video
        self.exp_vid = self.visualizer.visualize_on_video(
            rgb_score_map=self.rgb_score_map,
            alpha=self.vis_merged_dict['alpha']
        )

    def get_score_map(self, alpha_list=None, mean_type='harmonic'):
        """ Merges two or more score maps into one, using chosen mean and a
        weight for each dimension.

        Args:
            alpha_list (list): weights to use for the merge. One weight should
                be specified for each Xai object. Each weight should be a float
                between 0 and 1. If None, the weights will be set to the sum of
                the absolute values of the scores of each Xai object.
            mean_type (str, optional): type of mean to use to merge score maps.
                Can be one of: 'arithmetic', 'geometric' or 'harmonic'.
                Defaults to 'harmonic'.

        Returns:
            Score map containing merged explanation score maps.
        """

        # Init alpha_list if not given
        if alpha_list is None:
            # alpha_list = [1/len(self.xais) for _ in self.xais]
            alpha_list = [sum([t[1] for t in x.scores]) for x in self.xais]

        # Init score_arrays depending on type of mean
        if mean_type == 'geometric':
            pos_scores = np.ones(self.video.shape[:3], dtype='float16')
            neg_scores = np.ones(self.video.shape[:3], dtype='float16')
        else:
            pos_scores = np.zeros(self.video.shape[:3], dtype='float16')
            neg_scores = np.zeros(self.video.shape[:3], dtype='float16')

        # Init positive and negative masks
        neg_mask = np.ones(self.video.shape[:3])
        pos_mask = np.ones(self.video.shape[:3])

        # Show progress or not
        progress = list(zip(self.xais, alpha_list))
        progress = tqdm(progress) if self.show_progress else progress

        for xai, alpha in progress:

            if xai.vis_dict['type'] == 'posneg':
                visualizer = PosNegVisualizer(
                    video=self.video,
                    segments=xai.segments,
                    scores=xai.scores)
                score_map = visualizer.get_score_map(
                    th=xai.vis_dict['th'],
                    top_k=xai.vis_dict['top_k'],
                    min_accum=xai.vis_dict['min_accum'],
                    improve_background=xai.vis_dict['improve_background'],
                    pos_only=xai.vis_dict['pos_only'],
                    neg_only=xai.vis_dict['neg_only'],
                    weight_by_volume=xai.vis_dict['weight_by_volume'],
                    blur_mode=xai.vis_dict['blur_mode'],
                    blur_radius=xai.vis_dict['blur_radius'],
                    blur_sigma=xai.vis_dict['blur_sigma'])

            elif xai.vis_dict['type'] == 'heatmap':
                visualizer = HeatmapVisualizer(
                    video=self.video,
                    segments=xai.segments,
                    scores=xai.scores)
                score_map = visualizer.get_score_map(
                    th=xai.vis_dict['th'],
                    top_k=xai.vis_dict['top_k'],
                    min_accum=xai.vis_dict['min_accum'],
                    pos_only=xai.vis_dict['pos_only'],
                    neg_only=xai.vis_dict['neg_only'],
                    improve_background=xai.vis_dict['improve_background'],
                    weight_by_volume=xai.vis_dict['weight_by_volume'],
                    blur_mode=xai.vis_dict['blur_mode'],
                    blur_radius=xai.vis_dict['blur_radius'],
                    blur_sigma=xai.vis_dict['blur_sigma'])

            elif xai.vis_dict['type'] == 'convolutional':
                visualizer = ConvolutionalVisualizer(
                    video=self.video,
                    scores=xai.scores,
                    kernel_size=xai.vis_dict['kernel_size'],
                    stride=xai.vis_dict['stride'])
                score_map = visualizer.get_score_map(
                    th=xai.vis_dict['th'],
                    top_k=xai.vis_dict['top_k'],
                    min_accum=xai.vis_dict['min_accum'],
                    improve_background=xai.vis_dict['improve_background'])

            # Resizing
            if self.video.shape[:3] != score_map.shape:
                scale = [self.video.shape[i] / score_map.shape[i]
                         for i in range(len(score_map.shape))]
                score_map = ndimage.zoom(score_map, scale, order=1)

            # Get arrays of positive only and negative only values
            pos_values = score_map.copy()
            pos_values[pos_values < 0] = 0
            neg_values = -score_map.copy()
            neg_values[neg_values < 0] = 0

            # Aggregate positive and negative values separately, depending
            # on the type of mean being used
            if mean_type == 'arithmetic':
                pos_scores += pos_values * alpha
                neg_scores += neg_values * alpha
            elif mean_type == 'geometric':
                pos_scores *= pos_values ** alpha
                neg_scores *= neg_values ** alpha
            elif mean_type == 'harmonic':
                pos_scores[pos_values > 0] += alpha/pos_values[pos_values > 0]
                pos_mask[pos_values <= 0] = 0
                neg_scores[neg_values > 0] += alpha/neg_values[neg_values > 0]
                neg_mask[neg_values <= 0] = 0

        # Merge of positive and negative values into one array and division by
        # total depending on type of mean being used
        if mean_type == 'arithmetic':
            res = (pos_scores - neg_scores) / sum(alpha_list)
        elif mean_type == 'geometric':
            res = pos_scores ** sum(alpha_list) - neg_scores ** sum(alpha_list)
        elif mean_type == 'harmonic':
            pos_scores[pos_scores > 0] = \
                sum(alpha_list) / pos_scores[pos_scores > 0]
            pos_scores *= pos_mask
            neg_scores[neg_scores > 0] = \
                sum(alpha_list) / neg_scores[neg_scores > 0]
            neg_scores *= neg_mask
            res = pos_scores - neg_scores

        return res

    def run_pipeline(self, verbose=False):
        """ Run the whole explanation process.

        Args:
            verbose (bool, optional): whether to display messages about the
                current state of the explanation execution. Defaults to False.
        """

        # Segmentation
        if verbose:
            print('segmenting')
        self.segment()

        # Perturbation
        if verbose:
            print('perturbing')
        self.perturb()

        # Explanation
        if verbose:
            print('explaining')
        self.explain()

        # Visualization
        if verbose:
            print('visualizing')
        self.visualize()
