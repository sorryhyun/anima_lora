"""``path_pattern`` glob semantics live in ``anime_tools`` (curation side of
the split — shared by the training subsets and every curation stage). Since
anime_tools 0.7.3 the function lives in the package's one image walk,
``anime_tools._walk``; this module is the trainer's single import point so a
future upstream move touches one line."""

from anime_tools._walk import filter_paths_by_glob  # noqa: F401
