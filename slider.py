from openTSNE import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

class Slider():
    """
    Slider class for openTSNE
    """
    def __init__(self, num_slides = 60, use_previous_as_init = True, tsne_kwarg_list = None, min_exaggeration = 0.85, max_exaggeration = 30.0, verbose = True):
        """
        Initialize the Slider class

        :param num_slides: Number of slides to create
        :param use_previous_as_init: Whether to use the previous slide as initialization for the next slide
        :param tsne_kwarg_list: List of dictionaries with keyword arguments for the TSNE class. If None, a default list is created
        :param min_exaggeration: Minimum exaggeration for the slides
        :param max_exaggeration: Maximum exaggeration for the slides
        :param verbose: Whether to print progress
        """
        self.num_slides = num_slides
        self.use_previous_as_init = use_previous_as_init
        self.min_exaggeration = min_exaggeration
        self.max_exaggeration = max_exaggeration
        self.image_slides = None
        self.verbose = verbose

        #if no tsne_kwarg_list is given, create it
        if tsne_kwarg_list is None:
            tsne_kwarg_list = [{} for i in range(num_slides)]
            for i in range(num_slides):
                if use_previous_as_init and i > 0:
                    #When using previous as init, a lower number of iterations is sufficient
                    tsne_kwarg_list[i]['n_iter'] = 50

        #set the exaggeration for each slide if none is given. Default is logarithmically decreasing from max to min
        exags = np.logspace(np.log10(min_exaggeration), np.log10(max_exaggeration), num_slides)
        if 'exaggeration' not in tsne_kwarg_list[0]:
            for i in range(num_slides):
                tsne_kwarg_list[i]['exaggeration'] = exags[-i-1]

        #set the number of early exaggeration iterations to 0 if not given
        if 'early_exaggeration_iter' not in tsne_kwarg_list[0]:    
            for i in range(num_slides):
                tsne_kwarg_list[0]['early_exaggeration_iter'] = 0

        if use_previous_as_init:
            self.tsne_list =  [TSNE(**tsne_kwarg_list[0], verbose=verbose)]
        else: 
            self.tsne_list = [TSNE(**tsne_kwarg_list[i], verbose=verbose) for i in range(num_slides)]

        self.tsne_kwarg_list = tsne_kwarg_list
        self.embeddings = None

    def fit(self, X):
        """
        Fit the embeddings to the data

        :param X: Data to fit
        """
        if self.use_previous_as_init:
            self.embeddings = [self.tsne_list[0].fit(X)]
            for i in range(1, self.num_slides):
                self.embeddings.append(self.embeddings[i-1].optimize(**self.tsne_kwarg_list[i], verbose=self.verbose))
            self.embeddings = np.array(self.embeddings)
        else:
            self.embeddings = np.array([tsne.fit(X) for tsne in self.tsne_list])
    
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

    def _plot_embedding(self, embedding, size=0.3, color=None, cmap='viridis', bound_type='trimmed_cov', title=None, ax=None):
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
            fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(*embedding.T, s=size, c=color, cmap=cmap)
        ax.axis('off')
        ax.set_title(title, fontsize=20)

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
            bounds = [[mean_trimmed[0] - 3*np.sqrt(cov_trimmed[0,0]), mean_trimmed[0] + 3*np.sqrt(cov_trimmed[0,0])], 
                        [mean_trimmed[1] - 3*np.sqrt(cov_trimmed[1,1]), mean_trimmed[1] + 3*np.sqrt(cov_trimmed[1,1])]]
            
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])

        scale = self._get_scale(embedding, max_length=0.5)
        fontprops = fm.FontProperties(size=18)
        scalebar = AnchoredSizeBar(ax.transData,
                        scale, f'{scale}', 'lower center', 
                        pad=0.1,
                        color='black',
                        frameon=False,
                        size_vertical=0.005*(bounds[1][1] - bounds[1][0]),
                        fontproperties=fontprops)
        ax.add_artist(scalebar)

        return ax

    def save_slides(self, prefix = 'slide', suffix = '.png', save_path = 'plots/', size = 0.3, color = None, cmap = 'viridis', bound_type = 'trimmed_cov'):
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
        if bound_type == 'keep':
            bounds = [[self.embeddings[:, :, 0].min(), self.embeddings[:, :, 0].max()], [self.embeddings[:, :, 1].min(), self.embeddings[:, :, 1].max()]]
            bounds = 1.2*np.array(bounds)
        for i, embedding in enumerate(self.embeddings):
            fig = plt.figure(figsize=(10, 10))
            # ###
            # mean = np.mean(embedding, axis=0)
            # #remove 5% of the points that are furthest away from the mean
            # dist = np.linalg.norm(embedding - mean, axis=1)
            # dist_sorted = np.sort(dist)
            # dist_threshold = dist_sorted[int(0.95*len(dist_sorted))]
            # embedding_trimmed = embedding[dist < dist_threshold]
            # ###

            self._plot_embedding(embedding, size, color, cmap, bound_type, title = f'Exaggeration: {self.tsne_kwarg_list[i]["exaggeration"]:.1f}')

            #set the bounds of the plot 10% larger than the embedding
            plt.savefig(save_path + prefix + str(i) + suffix, bbox_inches='tight', pad_inches=0.5)
            plt.close(fig)

    def save_video(self, file_name = 'video.mp4', size = 0.3, color = None, cmap = 'viridis', bound_type = 'trimmed_cov'):
        """
        Save the slides as a video

        :param file_name: Name of the file to save the video to
        :param size: Size of the scatter points
        :param color: Color of the scatter points
        :param cmap: Colormap for the scatter points
        :param keep_bounds: Whether to keep the bounds the same for all slides. Uses the maximum bounds of all slides.
        """
        if bound_type == 'keep':
            bounds = [[self.embeddings[:, :, 0].min(), self.embeddings[:, :, 0].max()], [self.embeddings[:, :, 1].min(), self.embeddings[:, :, 1].max()]]
            bounds = 1.2*np.array(bounds)
        fig = plt.figure(figsize=(10, 10))
        plt.axis('off')
        ax = plt.axes()
        if self.embeddings is None:
            print('No embeddings fitted yet')
            return

        def update(frame):
            if frame < self.num_slides:
                f = frame
            else:
                f = -(frame - self.num_slides + 1)

            embedding = self.embeddings[f]

            ax.clear()
            
            return self._plot_embedding(embedding, size, color, cmap, bound_type, title = f'Exaggeration: {self.tsne_kwarg_list[f]["exaggeration"]:.1f}')
        
        ani = animation.FuncAnimation(fig, update, frames=self.num_slides*2-1, interval=0, repeat=True, blit=False)
        ani.save(file_name, writer='ffmpeg', fps=9, dpi=300)