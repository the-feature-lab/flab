import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


def rcsetup(**kwargs):
    """Configure matplotlib rc settings for a consistent plot style.

    Args:
        dpi (int): Figure resolution in dots per inch. Default 120.
        panel_color (tuple): Axis facecolor as an RGB tuple. Default is white.
        fontsize (int): Base font size for title and axis text. Default 12.
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


class ExptTrace():
    """A dict-like container for recording experiment outcomes indexed by tuples
    of independent variable values. Measurements can be modified/retrieved by both
    direct bracket indexing (trace[a, b] = val) and keyword-based access via set/get.
    Measurement traces (along a variable axis) are retrieved using slicing and returned
    as a (possibly masked) ndarray ordered by sorted axis values.

    self.var_names is the list of names of experimental independent variables.
    A "config" is a tuple of values for each independent variable.
    An "outcome" is an experimental measurement, represented as a numeric scalar or array.
    Every outcome in a single ExptTrace must be of the same shape.

    Example:
        mse = ExptTrace(["trial", "ntrain", "ridge"])
        mse[0, 64, 0.1] = 0.42
        mse[0, 128, 0.1] = 0.55
        mse[1, 64, 0.1] = 0.31
        mse[:, 64, 0.1]   # → array([0.42, 0.31])
    """

    @classmethod
    def multi_init(cls, num_init, var_names):
        """Return a list of num_init independent ExptTrace instances."""
        return [cls(var_names) for _ in range(num_init)]

    def __init__(self, var_names):
        """
        Args:
            var_names (list of str): Names of the independent variables that
                together define a config. "outcome" is a reserved name and
                must not appear in this list.
        """
        if not isinstance(var_names, list):
            raise ValueError("var_names must be a list")
        if "outcome" in var_names:
            raise ValueError("variable name 'outcome' disallowed")
        self.var_names = var_names.copy()
        self._config2outcome = {}
        self.outcome_shape = None

    def __setitem__(self, key, val):
        """Record an outcome for a config. Key is a scalar or tuple of config values."""
        config, outcome = key, val
        # ensure config is a tuple of the correct length
        config = (config,) if not isinstance(config, tuple) else config
        if len(config) != len(self.var_names):
            raise ValueError(f"len config {len(config)} != num vars {len(self.var_names)}")
        # ensure config settings are of valid types
        allowed_types = (int, float, str, tuple, np.integer, np.floating)
        if not all(isinstance(c, allowed_types) for c in config):
            raise ValueError(f"config {config} elements must be one of {allowed_types}")
        # ensure config doesn't already exist, then write measurement outcome
        if config in self._config2outcome:
            raise ValueError(f"config {config} already exists. overwriting not supported")
        # if this is the first measurement, figure out shape of measurement outcome
        if self.outcome_shape is None:
            out_array = np.asarray(outcome)
            if not np.issubdtype(out_array.dtype, np.number):
                raise ValueError("measurement outcome must be numeric")
            self.outcome_shape = out_array.shape
        # otherwise, ensure new measurement has compatible shape
        elif np.shape(outcome) != self.outcome_shape:
            raise ValueError(f"outcome shape {np.shape(outcome)} != expected {self.outcome_shape}")
        self._config2outcome[config] = outcome

    def __getitem__(self, key):
        """Retrieve outcomes for one or more configs.

        Key is a scalar/tuple of config values or slice(None) per variable.
        A bare slice (:) for a variable selects all recorded values for that
        variable, returning an ndarray (or masked array if some configs are
        missing) with axes ordered by sorted variable values.

        Returns a squeezed plain ndarray for a single config, a plain ndarray
        if no values are missing, or a masked ndarray otherwise.

        Raises:
            KeyError: If none of the selected configs have been written.
        """
        # we need to know shape of measurement outcome
        if self.outcome_shape is None:
            raise RuntimeError("must add items before getting")
        # key = tuple of indexers (ints or slices). Selects configs.
        # ensure key is a tuple of the correct length
        var_indexers = (key,) if not isinstance(key, tuple) else key
        if len(var_indexers) != len(self.var_names):
            raise ValueError(f"num config vars {len(var_indexers)} != expected {len(self.var_names)}")

        # for each indep var, get the var value selected by the key.
        # if the indexer is a (full) slice, get the full axis for that var
        config_axes = []
        for idx, var_name in enumerate(self.var_names):
            var_setting = var_indexers[idx]
            config_axis = [var_setting]
            if isinstance(var_setting, slice):
                slc = (var_setting.start, var_setting.stop, var_setting.step)
                if not all([x is None for x in slc]):
                    raise ValueError(f"slice start/stop/step not supported ({var_name})")
                config_axis = self.get_axis(var_name)
            config_axes.append(config_axis)

        # create a meshgrid of all selected configs, populate with outcomes.
        # use masked array to handle missing/unwritten outcomes.
        config_shape = [len(ax) for ax in config_axes]
        result_mesh = np.ma.masked_all(config_shape + list(self.outcome_shape))
        for mesh_idxs in np.ndindex(*config_shape):
            config = tuple(config_axes[dim][idx] for dim, idx in enumerate(mesh_idxs))
            if config in self._config2outcome.keys():
                result_mesh[mesh_idxs] = self._config2outcome[config]

        # if all results are missing, raise KeyError.
        # if the key selects a single measurement, return a squeezed array.
        # if there are no missing results, return a regular ndarray.
        if np.all(result_mesh.mask):
            raise KeyError(f"config(s) {var_indexers} is/are missing")
        if np.prod(config_shape) == 1:
            return np.array(result_mesh).squeeze()
        if not np.ma.is_masked(result_mesh):
            return np.array(result_mesh)
        return result_mesh

    def __str__(self):
        shape_str = str(self.outcome_shape) if self.outcome_shape is not None else "unknown"
        vars_str = ", ".join(self.var_names) if self.var_names else "(none)"
        return f"ExptTrace(vars=[{vars_str}], outcome_shape={shape_str})"

    def get_axis(self, var_name):
        """Return the sorted list of all recorded values for a variable."""
        if var_name not in self.var_names:
            raise ValueError(f"var {var_name} not found")
        var_idx = self.var_names.index(var_name)
        # iterate through written configs and collect all var settings
        axis = set()
        for config in self._config2outcome.keys():
            axis.add(config[var_idx])
        return sorted(list(axis))

    def get(self, **kwargs):
        """Retrieve outcomes using keyword arguments for each variable.

        Unspecified variables are sliced in full (equivalent to [:]).
        """
        key = self._get_config_key(_mode='get', **kwargs)
        return self[key]

    def set(self, **kwargs):
        """Record an outcome using keyword arguments. Requires outcome=<value>."""
        if "outcome" not in kwargs:
            raise ValueError(f"no outcome given")
        outcome = kwargs["outcome"]
        config = self._get_config_key(_mode='set', **kwargs)
        self[config] = outcome

    def is_written(self, **kwargs):
        """Return True if the given config (specified by kwargs) has been recorded."""
        config = self._get_config_key(_mode='set', **kwargs)
        return config in self._config2outcome.keys()

    def _get_config_key(self, _mode='set', **kwargs):
        key = []
        for var_name in self.var_names:
            var_indexer = kwargs.get(var_name, None)
            if var_indexer is None:
                if _mode == 'set':
                    raise ValueError(f"must specify var {var_name}")
                var_indexer = slice(None)  # full slice indexer in “get” mode
            key.append(var_indexer)
        return tuple(key)

    def serialize(self):
        """Return a plain dict representation suitable for pickling or JSON storage."""
        return {
            "var_names": self.var_names,
            "config2outcome": self._config2outcome,
            "outcome_shape": self.outcome_shape
        }

    @classmethod
    def deserialize(cls, data):
        """Reconstruct an ExptTrace from a dict produced by serialize()."""
        try:
            obj = cls(data["var_names"])
            obj._config2outcome = data["config2outcome"]
            obj.outcome_shape = data["outcome_shape"]
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
