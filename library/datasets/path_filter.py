"""``path_pattern`` glob semantics live in ``anime_tools._walk`` (shared by the
training subsets and every curation stage); this module is the trainer's single
import point."""

from anime_tools._walk import filter_paths_by_glob  # noqa: F401
