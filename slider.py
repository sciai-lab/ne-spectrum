from openTSNE.tsne import TSNE, TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

class Slider():
    #reST docstring
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

    def _plot_embedding(self, embedding, size = 0.3, color = None, cmap = 'viridis', bound_type = 'trimmed_cov', title = None):
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
        plt.scatter(*embedding.T, s=size, c=color, cmap=cmap)
        plt.axis('off')
        plt.title(title, fontsize=25)

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
            
        plt.xlim(bounds[0])
        plt.ylim(bounds[1])

        scale = self._get_scale(embedding, max_length=0.5)
        fontprops = fm.FontProperties(size=18)
        ax = plt.gca()
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

    def save_video(self, file_name = 'video.mp4', size = 0.3, color = None, cmap = 'viridis', bound_type = 'max'):
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


    # def _change_plot(self, current_exaggeration, ax, plot, color, cmap):
    #     self.current_exaggeration = current_exaggeration
    #     if self.embeddings is None:
    #         print('No embeddings fitted yet')
    #         return
    #     #find the closest embedding to the current exaggeration
    #     exags = np.array([self.tsne_kwarg_list[i]['exaggeration'] for i in range(self.num_slides)])
    #     closest_index = np.argmin(np.abs(exags - current_exaggeration))

    #     ax.set_title(f'Exaggeration: {self.tsne_kwarg_list[closest_index]["exaggeration"]}')
    #     bounds = [[self.embeddings[closest_index, :, 0].min(), self.embeddings[closest_index, :, 0].max()], 
    #               [self.embeddings[closest_index, :, 1].min(), self.embeddings[closest_index, :, 1].max()]]

    #     if self.fast_mode:
    #         embedding = self.embeddings[closest_index]
    #         if color is None:
    #             h = histogram2d(embedding[:, 0], embedding[:, 1], range=bounds, bins=300)
    #             im = ax.imshow(h, cmap=cmap)
    #             plt.title(f'Exaggeration: {self.exaggeration}')
    #             if im is None:
    #                 #im = ax.imshow(self.embedding[:, 0].reshape(499,499))
    #                 h = histogram2d(embedding[:, 0], embedding[:, 1], range=bounds, bins=300)
    #                 im = ax.imshow(h, cmap=cmap)
    #                 plt.title(f'Exaggeration: {self.exaggeration}')
    #             else:
    #                 #im.set_array(self.embedding[:, 0].reshape(499,499))
    #                 #im.autoscale()
    #                 h = histogram2d(embedding[:, 0], embedding[:, 1], range=bounds, bins=300)
    #                 im.set_array(h)
    #                 im.autoscale()
    #                 plt.title(f'Exaggeration: {self.exaggeration}')

    #         else:
    #             #get rgb colors for each point from the colormap
    #             colors = cmap(color)
    #             #create 3 histograms for each color channel
    #             h_r = histogram2d(embedding[:, 0], embedding[:, 1], range=bounds, bins=300, weights=colors[:,0])
    #             h_g = histogram2d(embedding[:, 0], embedding[:, 1], range=bounds, bins=300, weights=colors[:,1])
    #             h_b = histogram2d(embedding[:, 0], embedding[:, 1], range=bounds, bins=300, weights=colors[:,2])
    #             # #normalize the histograms to [0,1] by dividing by the maximum value of all histograms
    #             # maximum = np.max([h_r.max(), h_g.max(), h_b.max()])
    #             # h_r = h_r / maximum
    #             # h_g = h_g / maximum
    #             # h_b = h_b / maximum
    #             # normalize the histograms to [0,1] logarithmically but not separately for each channel
    #             maximum = np.max([h_r.max(), h_g.max(), h_b.max()])
    #             h_r = np.log(h_r + 1) / np.log(maximum + 1)
    #             h_g = np.log(h_g + 1) / np.log(maximum + 1)
    #             h_b = np.log(h_b + 1) / np.log(maximum + 1)

    #             #combine the histograms to a single image
    #             h = np.stack([h_r, h_g, h_b], axis=2)
    #             if plot is None:
    #                 plot = ax.imshow(h)
    #             else:
    #                 plot.set_array(h)
    #                 plot.autoscale()
                
    #             return plot
    #     else:
    #         #change position of the scatter points
    #         if plot is None:
    #             plot = ax.scatter(*self.embeddings[-1].T, s=0.3, c=color, cmap=cmap)
    #         else:
    #             plot.set_offsets(self.embeddings[closest_index])
    #         print(bounds)
    #         ax.set_xlim(bounds[0])
    #         ax.set_ylim(bounds[1])

    #         return plot
        
    # def show(self, color = None, cmap = 'viridis', size = 0.3):
    #     #show the embeddings with an interactive slider
    #     step_size = (self.max_exaggeration - self.min_exaggeration) / (self.num_slides-1)
    #     fig = plt.figure(figsize=(10, 10))
    #     plt.axis('off')
    #     ax = plt.axes()

    #     #get cmap from self.cmap
    #     _cmap = plt.get_cmap(cmap)
    #     _cmap.set_bad(_cmap.get_under())
        
    #     plot = self._change_plot(self.min_exaggeration, ax, None, color, _cmap)

    #     interact(lambda exaggeration: self._change_plot(exaggeration, ax, plot, color, _cmap), exaggeration = widgets.FloatSlider(min=self.min_exaggeration, max=self.max_exaggeration, step=step_size, value=self.min_exaggeration))
    #     plt.show()

    # def _precompute_images(self, color = None, cmap = 'viridis', size = 0.3):
    #     self.image_slides = []
    #     for i, embedding in enumerate(self.embeddings):
    #         fig = plt.figure(figsize=(10, 10))
    #         plt.axis('off')
    #         ax = plt.axes()
    #         ax.scatter(*embedding.T, s=size, c=color, cmap=cmap)
    #         plt.title(f'Exaggeration: {self.tsne_kwarg_list[i]["exaggeration"]}')
    #         #get image from figure

    #         self.image_slides.append(fig)
    #         plt.close(fig)

    # def _plot_precomputed(self, current_exaggeration, ax, plot):
    #     self.current_exaggeration = current_exaggeration

    #     #find the closest embedding to the current exaggeration
    #     exags = np.array([self.tsne_kwarg_list[i]['exaggeration'] for i in range(self.num_slides)])
    #     closest_index = np.argmin(np.abs(exags - current_exaggeration))

    #     ax.set_title(f'Exaggeration: {self.tsne_kwarg_list[closest_index]["exaggeration"]}')
    #     ax.clear()
    #     plot = ax.imshow(self.image_slides[closest_index].get_children()[0].get_children()[0].get_array())
    #     return plot
            
    # def show_precompute(self, color = None, cmap = 'viridis', size = 0.3):
    #     if self.image_slides is None:
    #         self._precompute_images(color, cmap, size)

    #     step_size = (self.max_exaggeration - self.min_exaggeration) / (self.num_slides-1)
    #     fig = plt.figure(figsize=(10, 10))
    #     plt.axis('off')
    #     ax = plt.axes()

    #     plot = self._plot_precomputed(self.min_exaggeration, ax, None)

    #     interact(lambda exaggeration: self._change_plot(exaggeration, ax, plot), exaggeration = widgets.FloatSlider(min=self.min_exaggeration, max=self.max_exaggeration, step=step_size, value=self.min_exaggeration))
        
    #     plt.show()
        

# class LivePlot():
#     def __init__(self, X, color = None, cmap = 'viridis', num_samples_to_plot = None):
#         self.exaggeration = 1
#         self.color = color
#         self.cmap = cmap
#         self.num_samples_to_plot = num_samples_to_plot
#         affinities = affinity.PerplexityBasedNN(
#             X,
#             perplexity=30,
#             metric="euclidean",
#             verbose=True,
#             n_jobs=-1
#         )
#         init = initialization.pca(X)
#         self.embedding = TSNEEmbedding(init, affinities, verbose=True)
#         self.paused = False

#     def change_exaggeration(self,exaggeration):
#         self.exaggeration = exaggeration

#     def show(self):
#         interact(self.change_exaggeration, exaggeration = widgets.FloatSlider(min=0.1, max=12, step=1, value=12))

#         fig = plt.figure(figsize=(10, 10))
#         plt.axis('off')
#         ax = plt.axes()

#         scatter = ax.scatter(*self.embedding[:self.num_samples_to_plot].T, s=1, c=self.color[:self.num_samples_to_plot], cmap=self.cmap)

#         def update(frame):
#             self.embedding.optimize(n_iter=5, exaggeration=self.exaggeration, momentum=0.8, inplace=True, n_jobs=-1)
#             scatter.set_offsets(self.embedding[:self.num_samples_to_plot])
#             ax.set_xlim(self.embedding[:, 0].min(), self.embedding[:, 0].max())
#             ax.set_ylim(self.embedding[:, 1].min(), self.embedding[:, 1].max())
#             plt.title(f'Exaggeration: {self.exaggeration}')
#             return scatter, 
    
#         def toggle_pause(*args, **kwargs):
#             if self.paused:
#                 self.animation.resume()
#             else:
#                 self.animation.pause()
#             self.paused = not self.paused

#         self.animation = animation.FuncAnimation(fig, update, frames=100, interval=100, repeat=True, blit=True)

#         fig.canvas.mpl_connect('button_press_event', toggle_pause)

#         plt.show()
#         return self.animation
    

# class FastLivePlot():
#     def __init__(self, X, color = None, cmap = 'viridis', num_samples_to_plot = None, plot_hook = None, plot_hook_axis = None):
#         self.exaggeration = 1
#         self.color = color
#         self.cmap = cmap
#         self.num_samples_to_plot = num_samples_to_plot
#         affinities = affinity.PerplexityBasedNN(
#             X,
#             perplexity=30,
#             metric="euclidean",
#             verbose=True,
#             n_jobs=32
#         )
#         init = initialization.pca(X)
#         self.embedding = TSNEEmbedding(init, affinities, verbose=True)
#         self.paused = False
#         self.last_bounds = None
#         self.plot_hook = plot_hook
#         self.plot_hook_axis = plot_hook_axis

#     def change_exaggeration(self,exaggeration):
#         self.exaggeration = exaggeration

#     def _plot_embedding(self, im, ax, cmap):
#         if self.last_bounds is not None:
#             new_bounds = [[self.embedding[:, 0].min(), self.embedding[:, 0].max()], [self.embedding[:, 1].min(), self.embedding[:, 1].max()]]
#             bounds = 0.3 * 1.2*np.array(new_bounds) + 0.7 * self.last_bounds
#         else:
#             bounds = [[self.embedding[:, 0].min(), self.embedding[:, 0].max()], [self.embedding[:, 1].min(), self.embedding[:, 1].max()]]
#             bounds = 1.2*np.array(bounds)

#         if self.color is None:
#             h = histogram2d(self.embedding[:, 0], self.embedding[:, 1], range=bounds, bins=300)
#             im = ax.imshow(h, cmap=cmap)
#             plt.title(f'Exaggeration: {self.exaggeration}')
#             if im is None:
#                 #im = ax.imshow(self.embedding[:, 0].reshape(499,499))
#                 h = histogram2d(self.embedding[:, 0], self.embedding[:, 1], range=bounds, bins=300)
#                 im = ax.imshow(h, cmap=cmap)
#                 plt.title(f'Exaggeration: {self.exaggeration}')
#             else:
#                 #im.set_array(self.embedding[:, 0].reshape(499,499))
#                 #im.autoscale()
#                 h = histogram2d(self.embedding[:, 0], self.embedding[:, 1], range=bounds, bins=300)
#                 im.set_array(h)
#                 im.autoscale()
#                 plt.title(f'Exaggeration: {self.exaggeration}')

#         else:
#             #get rgb colors for each point from the colormap
#             colors = cmap(self.color)
#             #create 3 histograms for each color channel
#             h_r = histogram2d(self.embedding[:, 0], self.embedding[:, 1], range=bounds, bins=400, weights=colors[:,0])
#             h_g = histogram2d(self.embedding[:, 0], self.embedding[:, 1], range=bounds, bins=400, weights=colors[:,1])
#             h_b = histogram2d(self.embedding[:, 0], self.embedding[:, 1], range=bounds, bins=400, weights=colors[:,2])
#             # #normalize the histograms to [0,1] by dividing by the maximum value of all histograms
#             # maximum = np.max([h_r.max(), h_g.max(), h_b.max()])
#             # h_r = h_r / maximum
#             # h_g = h_g / maximum
#             # h_b = h_b / maximum
#             # normalize the histograms to [0,1] logarithmically but not separately for each channel
#             maximum = np.max([h_r.max(), h_g.max(), h_b.max()])
#             h_r = np.log(h_r + 1) / np.log(maximum + 1)
#             h_g = np.log(h_g + 1) / np.log(maximum + 1)
#             h_b = np.log(h_b + 1) / np.log(maximum + 1)

#             #combine the histograms to a single image
#             h = np.stack([h_r, h_g, h_b], axis=2)
#             if im is None:
#                 im = ax.imshow(h)
#             else:
#                 im.set_array(h)
#                 im.autoscale()

#             plt.title(f'Exaggeration: {self.exaggeration}')

#         if self.plot_hook is not None:
#             self.plot_hook(self.plot_hook_axis, self.embedding)

#         self.last_bounds = bounds
            
#         return im

#     def show(self):
#         interact(self.change_exaggeration, exaggeration = widgets.FloatSlider(min=0.1, max=12, step=0.01, value=12))

#         #get cmap from self.cmap
#         cmap = plt.get_cmap(self.cmap)
#         cmap.set_bad(cmap.get_under())

#         fig = plt.figure(figsize=(10, 10))
#         plt.axis('off')
#         ax = plt.axes()

#         self.im = self._plot_embedding(None, ax, cmap)

#         def update(frame):
#             self.embedding.optimize(n_iter=10, exaggeration=self.exaggeration, momentum=0.8, inplace=True, n_jobs=32)

#             #ax.clear()
#             self.im = self._plot_embedding(self.im, ax, cmap)

#             return self.im, 
    
#         def toggle_pause(*args, **kwargs):
#             if self.paused:
#                 self.animation.resume()
#             else:
#                 self.animation.pause()
#             self.paused = not self.paused

#         self.animation = animation.FuncAnimation(fig, update, frames=100, interval=100, repeat=True, blit=True)

#         fig.canvas.mpl_connect('button_press_event', toggle_pause)

#         plt.show()
#         return self.animation