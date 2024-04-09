from openTSNE import TSNE
from cne import CNE
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from functools import partial

class Slider():
    """
    Slider class for openTSNE
    """
    iter_name = None
    early_exaggeration_iter_name = 0

    early_exaggeration_name = None

    spectrum_param_print_name = None
    spectrum_param_name = None

    def __init__(self,
                 num_slides=60,
                 use_previous_as_init=True,
                 early_exaggeration=0,
                 verbose=True,
                 kwarg_list=None,
                 min_spec_param=None,
                 max_spec_param=None,
                 **kwargs):

        """
        Initialize the Slider class

        :param num_slides: Number of slides to create
        :param use_previous_as_init: Whether to use the previous slide as initialization for the next slide
        :param tsne_kwarg_list: List of dictionaries with keyword arguments for the TSNE class. If None, a default list is created
        :param min_exaggeration: Minimum exaggeration for the slides
        :param max_exaggeration: Maximum exaggeration for the slides
        :param verbose: Whether to print progress inside each method. Overall progress over slides is always printed.
        :param early_exaggeration: Early exaggeration. If use_previous_as_init is True, this is only used in the first
         slide. For t-SNE this is an int for the number of early exaggeration epochs. For CNE this is a bool for whether
         to use early exaggeration.
        :param kwargs: Additional keyword arguments for the embedding method (TSNE or CNE class).
         Should be None, one dictionary, or a list with one dictionary for each slide. If None a default list is created.
         If one dictionary is given, this is used for all slides.
        :param min_spec_param: Minimum value for the spectrum parameter (exaggeration in t-SNE, s in CNE)
        :param max_spec_param: Maximum value for the spectrum parameter (exaggeration in t-SNE, s in CNE)
        """
        # store arguments
        self.num_slides = num_slides
        self.use_previous_as_init = use_previous_as_init
        self.verbose = verbose
        self.early_exaggeration = early_exaggeration
        self.kwarg_list = kwarg_list
        self.min_spec_param = min_spec_param
        self.max_spec_param = max_spec_param
        self.embeddings = None
        self.embedder_list = None

        # create and populate the list of kwargs for the slides
        set_iters = self._create_kwarg_list(kwargs)
        self._set_iterations(set_iters)
        self._set_verbosity()
        self._set_spectrum_param()
        self._set_early_exaggeration()

    def _set_early_exaggeration(self):
        """
        Set the early exaggeration for the slides
        """
        # set the number of early exaggeration iterations
        for i in range(self.num_slides):
            if self.use_previous_as_init:
                # there should not be any early exaggeration in the kwargs lists of the later slides!
                if i > 0 and self.early_exaggeration_name in self.kwarg_list[i]:
                    del self.kwarg_list[i][self.early_exaggeration_name]

                # if early exaggeration is given in the kwarg_list, this has priority
                if i == 0 and self.early_exaggeration_name not in self.kwarg_list[i]:
                    self.kwarg_list[i][self.early_exaggeration_name] = self.early_exaggeration

            else:
                # any slide might have early exaggeration
                # if early exaggeration is given in the kwarg_list, this has priority
                if self.early_exaggeration_name not in self.kwarg_list[i]:
                    self.kwarg_list[i][self.early_exaggeration_name] = self.early_exaggeration

    def _set_spectrum_param(self):
        # compute the intermediate spectrum parameters used for each slide
        self._get_intermediate_spectrum_params()

        # set the spectrum parameters for each slide if none is given.
        for i in range(self.num_slides):
            if self.spectrum_param_name not in self.kwarg_list[i]:
                self.kwarg_list[i][self.spectrum_param_name] = self.spectrum_params[i]

    def _get_intermediate_spectrum_params(self):
        """
        Get the intermediate spectrum parameters
        """
        self.spectrum_params = None

    def _set_verbosity(self):
        """
        Set the verbosity for the slides
        """
        pass

    def print_spectrum_param(self, i):
        """
        Print the spectrum parameter. Default printing, cannot be switched off.
        """
        print(f"Slide {i} / {self.num_slides} "
              f"with {self.spectrum_param_print_name}: {np.round(self.kwarg_list[i][self.spectrum_param_name], 2)}")

    def _create_kwarg_list(self, kwargs):
        """
        Create a list of kwargs for the slides. If no kwarg_list exists and new one is created without any items.
        If a single dictionary is given, this is used for all slides. If additional kwargs are given in the constructor,
        these are added to all slides, if not already present in the dictionary.

        """
        # do iterations have to be set?
        set_iters = False

        # create list of kwarg dictionaries, one for each embedder
        if self.kwarg_list is None:
            self.kwarg_list = [{} for _ in range(self.num_slides)]
            set_iters = True
        elif type(self.kwarg_list) == dict:
            self.kwarg_list = [self.kwarg_list.copy() for _ in range(self.num_slides)]
            set_iters = True
        else:
            # here we assume that iterations are already set
            assert len(self.kwarg_list) == self.num_slides, \
                "When passing a list for kwarg_list, its length must be equal to num_slides."

        # add the additional kwarg arguments, unless the key is already present in the dictionary
        for i in range(self.num_slides):
            for key, value in kwargs.items():
                if key not in self.kwarg_list[i]:
                    self.kwarg_list[i][key] = value
        return set_iters

    def _set_iterations(self, set_iters=False):
        """
        Set the number of iterations for the slides. If use_previous_as_init is True, the number of iterations
        is reduced for the subsequent slides to 50 iterations. Otherwise, we stick to the default number of iterations.
        """
        if set_iters:
            for i in range(self.num_slides):
                if self.use_previous_as_init and i > 0:
                    # When using previous as init, a lower number of iterations is sufficient
                    self.kwarg_list[i][self.iter_name] = 50

    def save_embeddings(self, file_name = 'embeddings.npy'):
        """
        Save the embeddings to a file

        :param file_name: Name of the file to save the embeddings to
        """
        np.save(file_name, self.embeddings)
    
    def load_embeddings(self, file_name = 'embeddings.npy'):
        """
        Load the embeddings from a file

        :param file_name: Name of the file to load the embeddings from
        """
        self.embeddings = np.load(file_name)
    
    def get_embeddings(self):
        """
        Return the embeddings

        :returns embeddings: The embeddings
        """
        return self.embeddings
    
    def _get_scale(self, embedding, max_length=0.5):
        """
        Return the smallest power of 10 that is smaller than max_length * the 
        spread in the x direction
        
        :returns scale: The scale
        """
        spreads = embedding.max(0) - embedding.min(0)
        spread = spreads.max()

        return 10 ** (int(np.log10(spread * max_length)))

    def _plot_embedding(self,
                        embedding,
                        size=2.0,
                        color=None,
                        cmap='viridis',
                        bound_type='trimmed_cov',
                        title=None,
                        ax=None,
                        scalebar=True):
        """
        Plot the embedding. Returns the axis of the plot.

        :param embedding: The embedding to plot
        :param size: Size of the scatter points
        :param color: Color of the scatter points
        :param cmap: Colormap for the scatter points
        :param bound_type: Type of bounds to use. Can be 'max' or 'trimmed_cov'
        :param title: Title of the plot

        :returns ax: The axis of the plot
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)

        # if no color values are passed, cmap is pointless and we want to avoid the warning
        if not (isinstance(color, np.ndarray) and color.dtype.kind in "if"):
            cmap = None
        ax.scatter(*embedding.T, s=size, c=color, cmap=cmap, edgecolor="none")
        ax.axis('off')
        ax.set_title(title, fontsize=10)

        if bound_type == 'max':
            bounds = [[embedding[:, 0].min(), embedding[:, 0].max()], [embedding[:, 1].min(), embedding[:, 1].max()]]
            bounds = 1.2*np.array(bounds)
        elif bound_type == 'trimmed_cov':
            #get mean
            mean = np.mean(embedding, axis=0)
            #remove 5% of the points that are furthest away from the mean
            dist = np.linalg.norm(embedding - mean, axis=1)
            dist_sorted = np.sort(dist)
            dist_threshold = dist_sorted[int(0.95*len(dist_sorted))]
            embedding_trimmed = embedding[dist < dist_threshold]

            #get trimmed mean and covariance
            mean_trimmed = np.mean(embedding_trimmed, axis=0)
            cov_trimmed = np.cov(embedding_trimmed.T)

            #get bounds from covariance
            bounds = [[mean_trimmed[0] - 2.5*np.sqrt(cov_trimmed[0, 0]), mean_trimmed[0] + 2.5*np.sqrt(cov_trimmed[0, 0])],
                      [mean_trimmed[1] - 2.5*np.sqrt(cov_trimmed[1, 1]), mean_trimmed[1] + 2.5*np.sqrt(cov_trimmed[1, 1])]]
            bounds = np.array(bounds)

        bound_diff = bounds[:, 1] - bounds[:, 0]

        if bound_diff[0] > bound_diff[1]:
            ax.set_xlim(bounds[0])
            ax.set_ylim([np.mean(bounds[1])-bound_diff[0]/2, np.mean(bounds[1])+bound_diff[0]/2])
        else:
            ax.set_ylim(bounds[1])
            ax.set_xlim([np.mean(bounds[0])-bound_diff[1]/2, np.mean(bounds[0])+bound_diff[1]/2])

        ylims = ax.get_ylim()

        ax.set_aspect('equal', "box")

        scale = self._get_scale(embedding, max_length=0.5)
        fontprops = fm.FontProperties(size=9)

        if scalebar:
            scalebar = AnchoredSizeBar(ax.transData,
                            scale, f'{scale}', 'lower center',
                            pad=0.1,
                            color='black',
                            frameon=False,
                            size_vertical=0.005*(ylims[1] - ylims[0]),
                            fontproperties=fontprops)
            ax.add_artist(scalebar)

        return ax

    def save_slides(self,
                    prefix='slide',
                    suffix='.png',
                    save_path='plots/',
                    size=2.0,
                    color=None,
                    cmap='viridis',
                    bound_type='trimmed_cov',
                    title=None):
        """
        Save the slides to a folder

        :param prefix: Prefix for the file names
        :param suffix: Suffix for the file names
        :param save_path: Path to the folder to save the slides to
        :param size: Size of the scatter points
        :param color: Color of the scatter points
        :param cmap: Colormap for the scatter points
        :param keep_bounds: Whether to keep the bounds the same for all slides. Uses the maximum bounds of all slides.
        """
        if os.path.isdir(save_path) == False:
            os.makedirs(save_path)
        for i, embedding in enumerate(self.embeddings):
            fig, ax = plt.subplots(figsize=(6.5, 6.5))

            if title is None:
                title = f'{self.spectrum_param_print_name}: {self.kwarg_list[i][self.spectrum_param_name]:.1f}'

            self._plot_embedding(embedding,
                                 size,
                                 color,
                                 cmap,
                                 bound_type,
                                 title=title,
                                 ax=ax)

            fig.savefig(os.path.join(save_path, prefix + str(i) + suffix))
            plt.close(fig)

    def save_video(self, save_path, file_name='video.mp4', size=2.0, color=None, cmap='viridis', bound_type='trimmed_cov', title=None, **kwargs):
        """
        Save the slides as a video
        #todo comment on gif vs mp4
        :param file_name: Name of the file to save the video to
        :param size: Size of the scatter points
        :param color: Color of the scatter points
        :param cmap: Colormap for the scatter points
        :param keep_bounds: Whether to keep the bounds the same for all slides. Uses the maximum bounds of all slides.
        """

        fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)

        if self.embeddings is None:
            print('No embeddings fitted yet')
            return

        def update(frame, title=None, ax=ax):
            if frame < self.num_slides:
                f = frame
            else:
                f = -(frame - self.num_slides + 1)

            embedding = self.embeddings[f]

            ax.clear()

            if title is None:
                title = f"{self.spectrum_param_print_name}: {self.kwarg_list[f][self.spectrum_param_name]:.1f}"

            return self._plot_embedding(embedding,
                                        size,
                                        color,
                                        cmap,
                                        bound_type,
                                        title=title,
                                        ax=ax,
                                        **kwargs)

        update = partial(update, title=title, ax=ax)

        ani = animation.FuncAnimation(fig, update, frames=self.num_slides*2-1, interval=0, repeat=True, blit=False)

        os.makedirs(save_path, exist_ok=True)
        ani.save(os.path.join(save_path, file_name), writer='ffmpeg', fps=9, dpi=300)
        plt.close(fig)



class TSNESlider(Slider):
    iter_name = "n_iter"
    embedder_class = TSNE
    spectrum_param_name = 'exaggeration'
    spectrum_param_print_name = 'Exaggeration'
    early_exaggeration_name = 'early_exaggeration_iter'
    def __init__(self,
                 num_slides=60,
                 use_previous_as_init=True,
                 early_exaggeration=0,
                 kwarg_list=None,
                 min_exaggeration=0.85,  # todo add a more abstract spectrum argument
                 max_exaggeration=30.0,
                 verbose=True,
                 **kwargs):
        super().__init__(num_slides=num_slides,
                         use_previous_as_init=use_previous_as_init,
                         verbose=verbose,
                         early_exaggeration=early_exaggeration,
                         kwarg_list=kwarg_list,
                         min_spec_param=min_exaggeration,
                         max_spec_param=max_exaggeration,
                         **kwargs
                         )
        # create embedders according to the kwarg_list
        self.create_embedder_list()

    def _get_intermediate_spectrum_params(self):
        # for t-SNE, the spectrum parameter is the exaggeration. We want to have a logarithmically decreasing spectrum
        self.spectrum_params = np.logspace(np.log10(self.min_spec_param),
                                           np.log10(self.max_spec_param),
                                           self.num_slides)[::-1]

    def create_embedder_list(self):
        """
        Create a list of embedders
        """
        if self.use_previous_as_init:
            self.embedder_list = [self.embedder_class(**self.kwarg_list[0])]
        else:
            self.embedder_list = [self.embedder_class(**self.kwarg_list[i]) for i in range(self.num_slides)]


    def _set_verbosity(self):
        # set verbosity
        for i in range(self.num_slides):
            if "verbose" not in self.kwarg_list[i]:
                self.kwarg_list[i]["verbose"] = self.verbose

    def fit(self, X):
        """
        Fit the embeddings to the data

        :param X: Data to fit
        """

        if self.use_previous_as_init:
            self.print_spectrum_param(0)
            self.embeddings = [self.embedder_list[0].fit(X)]
            for i in range(1, self.num_slides):
                self.print_spectrum_param(i)
                self.embeddings.append(self.embeddings[i-1].optimize(n_iter=self.kwarg_list[i]["n_iter"],
                                                                     exaggeration=self.kwarg_list[i]["exaggeration"],
                                                                     )
                                       )
        else:
            self.embeddings = []
            for i, embedder in enumerate(self.embedder_list):
                self.print_spectrum_param(i)
                self.embeddings.append(embedder.fit(X))
        self.embeddings = np.array(self.embeddings)


class CNESlider(Slider):
    iter_name = "n_epochs"
    embedder_class = CNE
    spectrum_param_name = 's'
    spectrum_param_print_name = 'Spectrum parameter'
    early_exaggeration_name = 'early_exaggeration'

    def __init__(self,
                 num_slides=60,
                 use_previous_as_init=True,
                 min_spec_param=-0.1,  # 0 is tsne, 1 is UMAP
                 max_spec_param=2.0,
                 kwarg_list=None,
                 verbose=True,
                 warmup=False,
                 overall_decay=None,
                 **kwargs):
        super().__init__(num_slides=num_slides,
                         use_previous_as_init=use_previous_as_init,
                         verbose=verbose,
                         kwarg_list=kwarg_list,
                         min_spec_param=min_spec_param,
                         max_spec_param=max_spec_param,
                            **kwargs)


        # add arguments for learning rate schedule to kwarg_list
        self.warmup = warmup
        self.overall_decay = overall_decay
        self._set_learning_rate()

        # create embedders according to the kwarg_list
        self.create_embedder_list()

    def _get_intermediate_spectrum_params(self):
        # for CNE, the spectrum parameter is "s". We want to have a linearly increasing spectrum as this is in log space
        # already
        self.spectrum_params = np.linspace(self.min_spec_param,
                                           self.max_spec_param,
                                           self.num_slides)[::-1]

    def _set_learning_rate(self):

        if self.warmup:
            warmup_share = 0.25

            for i in range(self.num_slides):
                # if a number of epochs is given, use a share of these for warmup, otherwise use a small default number
                if self.iter_name in self.kwarg_list[i]:
                    iter_slide = self.kwarg_list[i][self.iter_name]
                    warmup_slide = int(warmup_share * iter_slide)
                    self.kwarg_list[i]['warmup_epochs'] = warmup_slide
                else:
                    self.kwarg_list[i]['warmup_epochs'] = 10

                self.kwarg_list[i]['warmup_lr'] = 0.01

        if self.overall_decay == "linear":
            for i in range(self.num_slides):
                self.kwarg_list[i]['learning_rate'] = (1 - i / self.num_slides)
        elif self.overall_decay == "half_linear":
            for i in range(1, self.num_slides):
                self.kwarg_list[i]['learning_rate'] = 0.5 * (1 - (i-1) / (self.num_slides-1))
        elif self.overall_decay == "quarter_linear":
            for i in range(1, self.num_slides):
                self.kwarg_list[i]['learning_rate'] = 0.25 * (1 - (i-1) / (self.num_slides-1))
        elif self.overall_decay == "quarter":
            for i in range(1, self.num_slides):
                self.kwarg_list[i]['learning_rate'] = 0.25

    def _set_verbosity(self):
        # set verbosity
        for i in range(self.num_slides):
            if "print_freq_epoch" not in self.kwarg_list[i]:
                if self.verbose:
                    self.kwarg_list[i]["print_freq_epoch"] = "auto"
                else:
                    self.kwarg_list[i]["print_freq_epoch"] = None

    def create_embedder_list(self):
        """
        Create a list of embedders
        """
        self.embedder_list = [self.embedder_class(**self.kwarg_list[i]) for i in range(self.num_slides)]

    def fit(self, X):
        """
        Fit the embeddings to the data

        :param X: Data to fit
        """

        if self.use_previous_as_init:
            self.print_spectrum_param(0)
            self.embeddings = [self.embedder_list[0].fit_transform(X)]
            graph = self.embedder_list[0].neighbor_mat
            for i in range(1, self.num_slides):
                self.print_spectrum_param(i)
                self.embeddings.append(self.embedder_list[i].fit_transform(X, graph=graph, init=self.embeddings[i-1]))
        else:
            self.embeddings = []
            for i, embedder in enumerate(self.embedder_list):
                self.print_spectrum_param(i)
                self.embeddings.append(embedder.fit_transform(X))
        self.embeddings = np.array(self.embeddings)






