"""Utility modules with no current consumer. See attic/README.md.

This was ``src/utils/``. It moved here on 2026-07-26 after an audit found no
importer anywhere in the repo except ``scripts/diagnostics/test_system.py``,
which imported the names but never exercised them.

``AethalometerPlotter`` additionally rivals the sanctioned research plotting
package (``research/ftir_hips_chem/scripts/plotting/``), which covers five of
its six methods and is what all 31 active notebooks actually use. Its
constructor calls ``plt.style.use()``, which silently reverts the white
background ``apply_default_style()`` installs -- the hazard AGENTS.md warns
about. Do not wire it back into a research notebook.
"""

from .file_io import (
    save_results_to_json, load_results_from_json,
    save_dataframe_to_csv, ensure_output_directory,
)
from .memory_optimization import (
    MemoryOptimizer, BatchProcessor, CacheManager,
    memory_optimizer, batch_processor, cache_manager,
    optimize_memory, reduce_memory, process_in_batches,
)
from .logging.logger import ETADLogger
from .plotting import AethalometerPlotter
from .statistics import StatisticalAnalyzer

__all__ = [
    # File I/O
    'save_results_to_json', 'load_results_from_json',
    'save_dataframe_to_csv', 'ensure_output_directory',

    # Memory optimization
    'MemoryOptimizer', 'BatchProcessor', 'CacheManager',
    'memory_optimizer', 'batch_processor', 'cache_manager',
    'optimize_memory', 'reduce_memory', 'process_in_batches',

    # Logging
    'ETADLogger',

    # Plotting / statistics
    'AethalometerPlotter', 'StatisticalAnalyzer',
]
