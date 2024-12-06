from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox, TransformedBbox

import numbers

class RobustNormalize(matplotlib.colors.Normalize):
    def __init__(self, quantiles=None, vmin=None, vmax=None, center=False, clip=False):
        self.center = center
        if quantiles is None:
            self.q_max = .99
            self.q_min = .01
        elif isinstance(quantiles, numbers.Number):
            self.q_max = 1 - quantiles
            self.q_min = quantiles
        else:
            self.q_min, self.q_max = sorted(quantiles)
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        
    def autoscale_None(self, A):
        """If vmin or vmax are not set, use the min/max of *A* to set them."""
        A = np.asanyarray(A)
        if self.vmin is None and A.size:
            self.vmin = np.nanquantile(A,self.q_min)
        if self.vmax is None and A.size:
            self.vmax = np.nanquantile(A,self.q_max)
        if self.center:
            v = max([np.abs(self.vmin), np.abs(self.vmin)])
            self.vmin = -v
            self.vmax = v
            

class TransformableAxesImage(matplotlib.image.AxesImage):
    """A version of the AxesImage Artist that allows to transform its image"""
    
    def __init__(self, *args, datatransform=None,**kwargs):
        self._extent_internal = None
        super().__init__(*args,**kwargs)
        if datatransform is None:
            datatransform = matplotlib.transforms.Affine2D.identity()
        self._datatransform = datatransform
        self.set_transform(self._datatransform + self.axes.transData)
        
    
    def get_datatransform(self):
        return self._datatransform
    
    def set_datatransform(self, transform):
        self._datatransform = transform
        self.set_transform(self._datatransform + self.axes.transData)
        
    def get_extent(self):
        ext = super().get_extent()
        xmin, xmax , ymin, ymax = ext
        xy = np.array([[xmin,xmin,xmax,xmax],[ymin,ymax,ymax,ymin]]).T
        xy_trans = self.get_datatransform().transform(xy)
        return (np.min(xy_trans[:,0]), np.max(xy_trans[:,0]), np.min(xy_trans[:,1]), np.max(xy_trans[:,1]))
    
    def make_image(self, renderer, magnification=1.0, unsampled=False):
        # docstring inherited
        trans = self.get_transform()
        # image is created in the canvas coordinate.
        x1, x2, y1, y2 = super().get_extent()
        bbox = Bbox(np.array([[x1, y1], [x2, y2]]))
        transformed_bbox = TransformedBbox(bbox, trans)
        clip = ((self.get_clip_box() or self.axes.bbox) if self.get_clip_on()
                else self.figure.bbox)
        return self._make_image(self._A, bbox, transformed_bbox, clip,
                                magnification, unsampled=unsampled)
    
    
    @property
    def _extent(self):
        if self._extent_internal is not None:
            return self._extent_internal
        else:
            sz = self.get_size()
            numrows, numcols = sz
            if self.origin == 'upper':
                return (-0.5, numcols-0.5, numrows-0.5, -0.5)
            else:
                return (-0.5, numcols-0.5, -0.5, numrows-0.5)
    
    @_extent.setter
    def _extent(self, extent):
        self._extent_internal = extent
    
        

def imshow_transformable(X, cmap=None, norm=None, aspect=None,
               interpolation=None, alpha=None, vmin=None, vmax=None,
               origin=None, extent=None, *, filternorm=True, filterrad=4.0,
               resample=None, url=None, datatransform=None,ax=None, **kwargs):
        """
        Display data as a TransformableAxesImage, i.e., on a 2D regular raster.

        The input may either be actual RGB(A) data, or 2D scalar data, which
        will be rendered as a pseudocolor image. For displaying a grayscale
        image set up the colormapping using the parameters
        ``cmap='gray', vmin=0, vmax=255``.

        The number of pixels used to render an image is set by the Axes size
        and the *dpi* of the figure. This can lead to aliasing artifacts when
        the image is resampled because the displayed image size will usually
        not match the size of *X* (see
        :doc:`/gallery/images_contours_and_fields/image_antialiasing`).
        The resampling can be controlled via the *interpolation* parameter
        and/or :rc:`image.interpolation`.

        Parameters
        ----------
       
        
        X : array-like or PIL image
            The image data. Supported array shapes are:

            - (M, N): an image with scalar data. The values are mapped to
              colors using normalization and a colormap. See parameters *norm*,
              *cmap*, *vmin*, *vmax*.
            - (M, N, 3): an image with RGB values (0-1 float or 0-255 int).
            - (M, N, 4): an image with RGBA values (0-1 float or 0-255 int),
              i.e. including transparency.

            The first two dimensions (M, N) define the rows and columns of
            the image.

            Out-of-range RGB(A) values are clipped.

        cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
            The Colormap instance or registered colormap name used to map
            scalar data to colors. This parameter is ignored for RGB(A) data.

        norm : `~matplotlib.colors.Normalize`, optional
            The `.Normalize` instance used to scale scalar data to the [0, 1]
            range before mapping to colors using *cmap*. By default, a linear
            scaling mapping the lowest value to 0 and the highest to 1 is used.
            This parameter is ignored for RGB(A) data.

        aspect : {'equal', 'auto'} or float, default: :rc:`image.aspect`
            The aspect ratio of the Axes.  This parameter is particularly
            relevant for images since it determines whether data pixels are
            square.

            This parameter is a shortcut for explicitly calling
            `.Axes.set_aspect`. See there for further details.

            - 'equal': Ensures an aspect ratio of 1. Pixels will be square
              (unless pixel sizes are explicitly made non-square in data
              coordinates using *extent*).
            - 'auto': The Axes is kept fixed and the aspect is adjusted so
              that the data fit in the Axes. In general, this will result in
              non-square pixels.

        interpolation : str, default: :rc:`image.interpolation`
            The interpolation method used.

            Supported values are 'none', 'antialiased', 'nearest', 'bilinear',
            'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite',
            'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell',
            'sinc', 'lanczos', 'blackman'.

            If *interpolation* is 'none', then no interpolation is performed
            on the Agg, ps, pdf and svg backends. Other backends will fall back
            to 'nearest'. Note that most SVG renderers perform interpolation at
            rendering and that the default interpolation method they implement
            may differ.

            If *interpolation* is the default 'antialiased', then 'nearest'
            interpolation is used if the image is upsampled by more than a
            factor of three (i.e. the number of display pixels is at least
            three times the size of the data array).  If the upsampling rate is
            smaller than 3, or the image is downsampled, then 'hanning'
            interpolation is used to act as an anti-aliasing filter, unless the
            image happens to be upsampled by exactly a factor of two or one.

            See
            :doc:`/gallery/images_contours_and_fields/interpolation_methods`
            for an overview of the supported interpolation methods, and
            :doc:`/gallery/images_contours_and_fields/image_antialiasing` for
            a discussion of image antialiasing.

            Some interpolation methods require an additional radius parameter,
            which can be set by *filterrad*. Additionally, the antigrain image
            resize filter is controlled by the parameter *filternorm*.

        alpha : float or array-like, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
            If *alpha* is an array, the alpha blending values are applied pixel
            by pixel, and *alpha* must have the same shape as *X*.

        vmin, vmax : float, optional
            When using scalar data and no explicit *norm*, *vmin* and *vmax*
            define the data range that the colormap covers. By default,
            the colormap covers the complete value range of the supplied
            data. It is deprecated to use *vmin*/*vmax* when *norm* is given.
            When using RGB(A) data, parameters *vmin*/*vmax* are ignored.

        origin : {'upper', 'lower'}, default: :rc:`image.origin`
            Place the [0, 0] index of the array in the upper left or lower
            left corner of the Axes. The convention (the default) 'upper' is
            typically used for matrices and images.

            Note that the vertical axis points upward for 'lower'
            but downward for 'upper'.

            See the :doc:`/tutorials/intermediate/imshow_extent` tutorial for
            examples and a more detailed description.

        extent : floats (left, right, bottom, top), optional
            The bounding box in data coordinates that the image will fill.
            The image is stretched individually along x and y to fill the box.

            The default extent is determined by the following conditions.
            Pixels have unit size in data coordinates. Their centers are on
            integer coordinates, and their center coordinates range from 0 to
            columns-1 horizontally and from 0 to rows-1 vertically.

            Note that the direction of the vertical axis and thus the default
            values for top and bottom depend on *origin*:

            - For ``origin == 'upper'`` the default is
              ``(-0.5, numcols-0.5, numrows-0.5, -0.5)``.
            - For ``origin == 'lower'`` the default is
              ``(-0.5, numcols-0.5, -0.5, numrows-0.5)``.

            See the :doc:`/tutorials/intermediate/imshow_extent` tutorial for
            examples and a more detailed description.

        filternorm : bool, default: True
            A parameter for the antigrain image resize filter (see the
            antigrain documentation).  If *filternorm* is set, the filter
            normalizes integer values and corrects the rounding errors. It
            doesn't do anything with the source floating point values, it
            corrects only integers according to the rule of 1.0 which means
            that any sum of pixel weights must be equal to 1.0.  So, the
            filter function must produce a graph of the proper shape.

        filterrad : float > 0, default: 4.0
            The filter radius for filters that have a radius parameter, i.e.
            when interpolation is one of: 'sinc', 'lanczos' or 'blackman'.

        resample : bool, default: :rc:`image.resample`
            When *True*, use a full resampling method.  When *False*, only
            resample when the output image is larger than the input image.

        url : str, optional
            Set the url of the created `.AxesImage`. See `.Artist.set_url`.
            
        ax : Axes to display in
        
        datatransform : matplotlib.transforms.Transform
           Set the transform from pixel indices to data coordinates applied to the image

        Returns
        -------
        `~matplotlib.image.AxesImage`

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.artist.Artist` properties
            These parameters are passed on to the constructor of the
            `.AxesImage` artist.

        See Also
        --------
        matshow : Plot a matrix or an array as an image.

        Notes
        -----
        Unless *extent* is used, pixel centers will be located at integer
        coordinates. In other words: the origin will coincide with the center
        of pixel (0, 0).

        There are two common representations for RGB images with an alpha
        channel:

        -   Straight (unassociated) alpha: R, G, and B channels represent the
            color of the pixel, disregarding its opacity.
        -   Premultiplied (associated) alpha: R, G, and B channels represent
            the color of the pixel, adjusted for its opacity by multiplication.

        `~matplotlib.pyplot.imshow` expects RGB images adopting the straight
        (unassociated) alpha representation.
        """
        
        if ax is None:
            _, ax = plt.gca()
        # ugly, but I copied the matplotlib method
        self = ax
        if aspect is None:
            aspect = matplotlib.rcParams['image.aspect']
        self.set_aspect(aspect)
        im = TransformableAxesImage(self, cmap=cmap, norm=norm, interpolation=interpolation, origin=origin, extent=extent,
                              filternorm=filternorm, filterrad=filterrad,
                              resample=resample,datatransform=datatransform, **kwargs)

        im.set_data(X)
        im.set_alpha(alpha)
        if im.get_clip_path() is None:
            # image does not already have clipping set, clip to axes patch
            im.set_clip_path(self.patch)
        #im._scale_norm(norm, vmin, vmax)
        im.set_url(url)

        # update ax.dataLim, and, if autoscaling, set viewLim
        # to tightly fit the image, regardless of dataLim.
        #im.set_extent(im.get_extent())

        self.add_image(im)
        self.relim()
        self.autoscale()
        return im
