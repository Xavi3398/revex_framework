class Pipeline(object):

    def __init__(self, seg, per, exp, vis):
        """Pipeline for video explanation.
        
        Args:
            seg (dict): segmentation parameters.
            per (dict): perturbation parameters.
            exp (dict): explanation parameters.
            vis (dict): visualization parameters.
        """

        self.seg = seg.copy()
        self.per = per.copy()
        self.exp = exp.copy()
        self.vis = vis.copy()
        
class ProposedMethods(object):

    VideoLIME = Pipeline(
        seg={
            'type': 'slic',
            'n_segments': 200,
            'compactness': 20,
            'spacing': [0.2, 1, 1]},
        per={
            'type': 'multiple',
            'hide_color': 'blur', 
            'blur_init_mode': '3d', 
            'blur_init_radius': 25,
            'blur_mode': '2d', 
            'blur_radius': 10,
            'num_samples': 1000,
            'p': 0.5},
        exp={
            'type': 'lime',
            'kernel': 'lime'},
        vis={
            'type': 'heatmap',
            'pos_only': True,
            'min_accum': 0.3,
            'improve_background': True,
            'blur_mode': '2d',
            'blur_radius': 10})

    VideoKernelSHAP = Pipeline(
        seg={
            'type': 'slic',
            'n_segments': 200,
            'compactness': 20,
            'spacing': [0.2, 1, 1]},
        per={
            'type': 'shap',
            'hide_color': 'blur', 
            'blur_init_mode': '3d', 
            'blur_init_radius': 25,
            'blur_mode': '2d', 
            'blur_radius': 10,
            'num_samples': 1000,
            'algorithm': 'kernel'},
        exp={
            'type': 'from_perturber'},
        vis={
            'type': 'heatmap',
            'pos_only': True,
            'min_accum': 0.3,
            'improve_background': True,
            'blur_mode': '2d',
            'blur_radius': 10})

    VideoRISE = Pipeline(
        seg={
            'type': 'rise',
            'n_seg': [4, 7, 7]},
        per={
            'type': 'multiple',
            'hide_color': 'blur', 
            'blur_init_mode': '3d', 
            'blur_init_radius': 25,
            'num_samples': 1000},
        exp={
            'type': 'mean'},
        vis={
            'type': 'heatmap',
            'min_accum': 0.3,
            'improve_background': True})

    VideoLOCO = Pipeline(
        seg={
            'type': 'slic',
            'n_segments': 200,
            'compactness': 20,
            'spacing': [0.2, 1, 1]},
        per={
            'type': 'single',
            'hide_color': 'blur', 
            'blur_init_mode': '3d', 
            'blur_init_radius': 25,
            'blur_mode': '2d', 
            'blur_radius': 10},
        exp={
            'type': 'difference'},
        vis={
            'type': 'heatmap',
            'pos_only': True,
            'min_accum': 0.3,
            'improve_background': True,
            'blur_mode': '2d',
            'blur_radius': 10})

    VideoUP = Pipeline(
        seg={
            'type': 'slic',
            'n_segments': 200,
            'compactness': 20,
            'spacing': [0.2, 1, 1]},
        per={
            'type': 'allbutone',
            'hide_color': 'blur', 
            'blur_init_mode': '3d', 
            'blur_init_radius': 10,
            'blur_mode': '2d', 
            'blur_radius': 10},
        exp={
            'type': 'difference',
            'invert': True},
        vis={
            'type': 'heatmap',
            'pos_only': True,
            'min_accum': 0.3,
            'improve_background': True,
            'blur_mode': '2d',
            'blur_radius': 10})

    VideoSOS = Pipeline(
        seg={
            'type': 'void'},
        per={
            'type': 'convolutional',
            'hide_color': 'blur', 
            'blur_init_mode': '3d', 
            'blur_init_radius': 25,
            'blur_mode': '2d', 
            'blur_radius': 10,
            'kernel_size': [4, 7, 7],
            'stride': [2, 2, 2]},
        exp={
            'type': 'value'},
        vis={
            'type': 'convolutional',
            'kernel_size': [4, 7, 7],
            'stride': [2, 2, 2],
            'min_accum': 0.3,
            'improve_background': True})

class SeparateSpaceTimeMethods(object):
    
    SpaceX_LIME_black = Pipeline(
        seg={
            'type': 'grid',
            'n_seg': [1, 1, 7]},
        per={
            'type': 'multiple',
            'hide_color': 0,
            'num_samples': 100},
        exp={
            'type': 'lime',
            'kernel': 'lime'},
        vis={
            'type': 'posneg',
            'pos_only': True,
            'min_accum': 0.3,
            'improve_background': True})
    
    SpaceY_LIME_black = Pipeline(
        seg={
            'type': 'grid',
            'n_seg': [1, 7, 1]},
        per={
            'type': 'multiple',
            'hide_color': 0,
            'num_samples': 100},
        exp={
            'type': 'lime',
            'kernel': 'lime'},
        vis={
            'type': 'posneg',
            'pos_only': True,
            'min_accum': 0.3,
            'improve_background': True})
    
    Time_LIME_black = Pipeline(
        seg={
            'type': 'grid',
            'n_seg': [4, 1, 1]},
        per={
            'type': 'multiple',
            'hide_color': 0,
            'num_samples': 50},
        exp={
            'type': 'lime',
            'kernel': 'lime'},
        vis={
            'type': 'posneg',
            'pos_only': True,
            'min_accum': 0.3,
            'improve_background': True})

    SpaceX_LIME_blur = Pipeline(
        seg={
            'type': 'grid',
            'n_seg': [1, 1, 7]},
        per={
            'type': 'multiple',
            'hide_color': 'blur', 
            'blur_init_mode': '3d', 
            'blur_init_radius': 25,
            'blur_mode': '2d', 
            'blur_radius': 10,
            'num_samples': 100,
            'p': 0.5},
        exp={
            'type': 'lime',
            'kernel': 'lime'},
        vis={
            'type': 'heatmap',
            'pos_only': True,
            'min_accum': 0.3,
            'improve_background': True,
            'blur_mode': '2d',
            'blur_radius': 10})
    
    SpaceY_LIME_blur = Pipeline(
        seg={
            'type': 'grid',
            'n_seg': [1, 7, 1]},
        per={
            'type': 'multiple',
            'hide_color': 'blur', 
            'blur_init_mode': '3d', 
            'blur_init_radius': 25,
            'blur_mode': '2d', 
            'blur_radius': 10,
            'num_samples': 100,
            'p': 0.5},
        exp={
            'type': 'lime',
            'kernel': 'lime'},
        vis={
            'type': 'heatmap',
            'pos_only': True,
            'min_accum': 0.3,
            'improve_background': True,
            'blur_mode': '2d',
            'blur_radius': 10})

    Time_LIME_blur = Pipeline(
        seg={
            'type': 'grid',
            'n_seg': [4, 1, 1]},
        per={
            'type': 'multiple',
            'hide_color': 'blur', 
            'blur_init_mode': '3d', 
            'blur_init_radius': 25,
            'blur_mode': '2d', 
            'blur_radius': 10,
            'num_samples': 50,
            'p': 0.5},
        exp={
            'type': 'lime',
            'kernel': 'lime'},
        vis={
            'type': 'heatmap',
            'pos_only': True,
            'min_accum': 0.3,
            'improve_background': True,
            'blur_mode': '2d',
            'blur_radius': 10})
