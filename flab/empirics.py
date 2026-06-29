import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


def rcsetup(**kwargs):
    """Configure matplotlib rc settings for a consistent plot style.

    **kwargs:
        dpi (int): Figure resolution in dots per inch. Default 120.
        panel_color (tuple): Axis facecolor as an RGB tuple. Default is white.
        fontsize (int): Base font size for title and axis text. Default 12.
        retina (bool): If True, use retina display format for inline plots. Default False.
    """
    dpi = kwargs.get("dpi", 120)
    panel_color = kwargs.get("panel_color", (1, 1, 1))
    if panel_color == "parchment":
        panel_color = (1, .99, .96)
    font_size = kwargs.get("fontsize", 12)
    plt.rc("figure", dpi=dpi, facecolor=(1, 1, 1))
    plt.rc("font", family='stixgeneral', size=font_size)
    plt.rc("axes", facecolor=panel_color, titlesize=font_size)
    plt.rc("mathtext", fontset='cm')
    # Use TrueType fonts in PDF
    plt.rc("pdf", fonttype=42)
    if kwargs.get("retina", False):
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        set_matplotlib_formats('retina')


class ExptTrace():
    """A dict-like container for recording experiment measurements indexed by tuples
    of independent variable values. Measurements can be modified/retrieved by both
    direct bracket indexing (trace[a, b] = val) and keyword-based access via set/get.
    Measurement traces (along a variable axis) are retrieved using slicing and returned
    as a (possibly masked) ndarray ordered by sorted axis values.

    self.var_names is the list of names of experimental independent variables.
    Each key is a tuple of values specifying each such independent variable.
    Each val is an experimental measurement, represented as a numeric scalar or array.
    Every val in an ExptTrace must be of the same shape.

    Example:
        mse = ExptTrace(["trial", "ntrain", "ridge"])
        mse[0, 64, 0.1] = 0.42
        mse[0, 128, 0.1] = 0.55
        mse[1, 64, 0.1] = 0.31
        mse[:, 64, 0.1]                   # → array([[[0.42]], [[0.31]]])
        mse.trace(ntrain=64, ridge=0.1)   # → ([0, 1], array([0.42, 0.31]))
    """

    @classmethod
    def multi_init(cls, num_init, var_names):
        """Return a list of num_init independent ExptTrace instances."""
        return [cls(var_names) for _ in range(num_init)]

    def __init__(self, var_names):
        """
        Args:
            var_names (list of str): Names of the independent variables that
                together define an expt configuration.
        """
        if not isinstance(var_names, list):
            raise ValueError("var_names must be a list")
        self.var_names = var_names.copy()
        self.measurements = {}
        self.val_shape = None

    def __setitem__(self, key, val):
        """Given a config `key` (scalar or tuple), record a measurement `val`."""
        # ensure key is a tuple of the correct length
        key = (key,) if not isinstance(key, tuple) else key
        if len(key) != len(self.var_names):
            raise ValueError(f"len key {len(key)} != num vars {len(self.var_names)}")
        # ensure key settings are of valid types
        allowed_types = (int, float, str, tuple, np.integer, np.floating)
        if not all(isinstance(c, allowed_types) for c in key):
            raise ValueError(f"key {key} elements must be one of {allowed_types}")
        # ensure key doesn't already exist, then write measurement
        if key in self.measurements.keys():
            raise ValueError(f"key {key} already exists. overwriting not supported")
        # if this is the first measurement, figure out shape of measurement
        if self.val_shape is None:
            out_array = np.asarray(val)
            if not np.issubdtype(out_array.dtype, np.number):
                raise ValueError("measurement must be numeric")
            self.val_shape = out_array.shape
        # otherwise, ensure new measurement has compatible shape
        elif np.shape(val) != self.val_shape:
            raise ValueError(f"measurement shape {np.shape(val)} != expected {self.val_shape}")
        self.measurements[key] = val

    def __getitem__(self, key):
        """Retrieve measurements for one or more configurations of
        independent experimental variables.

        `key` is a scalar/tuple of var_name values or slice(None) per variable.
        A bare slice (:) for a variable selects all recorded values for that
        variable, returning an ndarray (or masked array if some measurements are
        missing) with axes ordered by sorted variable values.

        Returns a squeezed ndarray if a single measurement is selected,
        a plain ndarray if many values selected with none missing,
        or a masked ndarray otherwise.

        Raises:
            KeyError: If none of the selected measurements have been written.
        """
        # we need to know shape of measurement
        if self.val_shape is None:
            raise RuntimeError("must add items before getting")
        # key = tuple of indexers (ints or slices). Selects experimental configurations.
        # ensure key is a tuple of the correct length
        key = (key,) if not isinstance(key, tuple) else key
        if len(key) != len(self.var_names):
            raise ValueError(f"num variables {len(key)} != expected {len(self.var_names)}")

        # for each independent variable, get the indexer.
        # if the indexer is a (full) slice, get the full axis for that variable
        # then, construct the axes of all selected configurations, in order of var_names.
        config_axes = []
        for idx, var_name in enumerate(self.var_names):
            key_idxr = key[idx]
            config_axis = [key_idxr]
            if isinstance(key_idxr, slice):
                slc = (key_idxr.start, key_idxr.stop, key_idxr.step)
                if not all([x is None for x in slc]):
                    raise ValueError(f"slice start/stop/step not supported ({var_name})")
                config_axis = self.get_axis(var_name)
            config_axes.append(config_axis)

        # create a meshgrid of all selected configurations, populate with measurements.
        # use masked array to handle missing/unwritten measurements.
        key_shape = [len(ax) for ax in config_axes]
        result_mesh = np.ma.masked_all(key_shape + list(self.val_shape))
        for mesh_idxs in np.ndindex(*key_shape):
            _key = tuple(config_axes[dim][idx] for dim, idx in enumerate(mesh_idxs))
            if _key in self.measurements.keys():
                result_mesh[mesh_idxs] = self.measurements[_key]

        # if all measurements are missing, raise KeyError.
        # if the key selects a single measurement, return a squeezed array.
        # if there are no missing measurements, return a regular ndarray.
        # otherwise, return a masked array.
        if np.all(result_mesh.mask):
            raise KeyError(f"key(s) {key} is/are missing")
        if np.prod(key_shape) == 1:
            return np.array(result_mesh).squeeze()
        if not np.ma.is_masked(result_mesh):
            return np.array(result_mesh)
        return result_mesh

    def __str__(self):
        shape_str = str(self.val_shape) if self.val_shape is not None else "unknown"
        vars_str = ", ".join(self.var_names) if self.var_names else "(none)"
        return f"ExptTrace(vars=[{vars_str}], val_shape={shape_str})"

    def get_axis(self, var_name):
        """Return the sorted list of all recorded values for a variable."""
        if var_name not in self.var_names:
            raise ValueError(f"var {var_name} not found")
        var_idx = self.var_names.index(var_name)
        # iterate through written measurements and collect all var settings
        axis = set()
        for key in self.measurements.keys():
            axis.add(key[var_idx])
        return sorted(list(axis))

    def get(self, **kwargs):
        """Retrieve measurements using keyword arguments for each variable.
        Unspecified variables are sliced in full (equivalent to [:]).
        """
        return self[self._get_key('get', **kwargs)]

    def get_trace(self, **kwargs):
        """Retrieve a 1D trace of measurements along a single variable axis.

        Exactly one variable must be left unspecified (or set to None); that
        variable becomes the trace axis (equivalent to a [:] slice), while every
        other variable must be pinned to a scalar value.

        As a special case, two variables may be left unspecified if one of them
        is named "trial". The other (the "var" axis) becomes the trace axis, and
        measurements are aggregated across the (possibly ragged) trial axis into
        a per-var-setting mean and standard deviation.

        Returns:
            (axis, result):
                `axis`: sorted ndarray of trace-variable values.
                `result`: with one unspecified variable, an ndarray of shape
                    (len(axis), *val_shape) of the corresponding measurements.
                    With "trial" additionally unspecified, a tuple
                    (result_mean, result_std), each an ndarray of shape
                    (len(axis), *val_shape). In all cases len(axis) == len(result).

        Raises:
            ValueError: If the unspecified variables don't match an allowed case.
            KeyError: If no selected measurement has been written.
        """
        missing = [v for v in self.var_names if kwargs.get(v, None) is None]
        reduce_trials = len(missing) == 2 and "trial" in missing
        if not (len(missing) == 1 or reduce_trials):
            raise ValueError(
                "trace requires exactly one unspecified variable (or two if one "
                f"is 'trial'), got {missing}")

        if not reduce_trials:
            axis = self.get_axis(missing[0])
            result = self.get(**kwargs)

            # collapse the singleton pinned dims to align result with the trace
            # axis, keeping the trace axis first and any outcome dims trailing.
            result = result.reshape((len(axis),) + self.val_shape)
            if np.ma.is_masked(result):
                # a config is missing iff its whole measurement is masked; drop
                # those configs (and axis values), return plain (unmasked) data.
                present = ~np.ma.getmaskarray(result).reshape(len(axis), -1).all(axis=1)
                axis = [a for a, keep in zip(axis, present) if keep]
                result = np.asarray(result[present])
            else:
                result = np.asarray(result)
            return np.array(axis), result

        # two unspecified axes: aggregate across the "trial" axis.
        var = missing[0] if missing[1] == "trial" else missing[1]
        axis = self.get_axis(var)
        trial_axis = self.get_axis("trial")
        result = self.get(**kwargs)

        # collapse the singleton pinned dims, isolating the (trial, var) block.
        # the two full dims keep their relative order from var_names.
        if self.var_names.index("trial") < self.var_names.index(var):
            result = result.reshape((len(trial_axis), len(axis)) + self.val_shape)
            trial_dim = 0
        else:
            result = result.reshape((len(axis), len(trial_axis)) + self.val_shape)
            trial_dim = 1

        # reduce the (possibly ragged) trial dim: masked trials are ignored, so
        # each var setting is averaged over only the trials it actually has.
        # ddof=0 keeps std well-defined even for var settings with a single trial.
        result = np.ma.asarray(result)
        result_mean = np.ma.mean(result, axis=trial_dim)
        result_std = np.ma.std(result, axis=trial_dim)

        # a var setting is missing iff it has no trials at all (fully masked);
        # drop those (and their axis values) to align result with the trace axis.
        present = ~np.ma.getmaskarray(result_mean).reshape(len(axis), -1).all(axis=1)
        axis = [a for a, keep in zip(axis, present) if keep]
        result_mean = np.asarray(result_mean[present])
        result_std = np.asarray(result_std[present])
        return np.array(axis), (result_mean, result_std)

    def set(self, val, /, **kwargs):
        """Record a measurement using keyword arguments."""
        key = self._get_key('set', **kwargs)
        self[key] = val

    def is_written(self, **kwargs):
        """Return True if the given configuration (specified by kwargs) has been recorded."""
        key = self._get_key('set', **kwargs)
        return key in self.measurements.keys()

    def _get_key(self, mode='set', /, **kwargs):
        key = []
        for var_name in self.var_names:
            key_idxr = kwargs.get(var_name, None)
            if key_idxr is None:
                if mode == 'set':
                    raise ValueError(f"must specify var {var_name}")
                key_idxr = slice(None)
            key.append(key_idxr)
        return tuple(key)

    def serialize(self):
        """Return a plain dict representation suitable for pickling or JSON storage."""
        return {
            "var_names": self.var_names,
            "measurements": self.measurements,
            "val_shape": self.val_shape
        }

    @classmethod
    def deserialize(cls, data):
        """Reconstruct an ExptTrace from a dict produced by serialize()."""
        try:
            obj = cls(data["var_names"])
            obj.measurements = data["measurements"]
            obj.val_shape = data["val_shape"]
        except KeyError as e:
            raise ValueError(f"Missing key in serialized data: {e}")
        return obj


class FileManager():

    def __init__(self, root):
        """
        root (str): The root directory from which this FileManager works.
        """
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.filepath = self.root

    def set_filepath(self, *paths):
        """
        Set the current filepath relative to the root directory. Helpful for temporarily
        going into a subdirectory.

        *paths (str): Variable number of path components to join.
        """
        self.filepath = os.path.join(self.root, *paths)
        os.makedirs(self.filepath, exist_ok=True)

    def get_filename(self, fn):
        """
        Get the absolute file path given a filename relative to the current filepath.
        fn (str): The filename relative to the current filepath.
        """
        return os.path.join(self.filepath, fn)

    def save(self, obj, fn):
        """
        Store an object to disk.

        obj (object): The object to be saved.
        fn (str): The filename relative to the current filepath. Should end in .npy if obj is ndarray.
        """
        fn = self.get_filename(fn)
        if fn.endswith('.npy'):
            assert isinstance(obj, np.ndarray)
            np.save(fn, obj)
            return
        with open(fn, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, fn):
        """
        Load an object from disk.

        fn (str): The filename relative to the current filepath.
        Returns: The loaded object, or None if the file does not exist.
        """
        fn = self.get_filename(fn)
        if not os.path.isfile(fn):
            return None
        if fn.endswith('.npy'):
            obj = np.load(fn)
            return obj
        with open(fn, 'rb') as handle:
            obj = pickle.load(handle)
        return obj
