"""Optional CF output gating by distinct users in the frozen training graph."""
from __future__ import annotations

import numpy as np

from recall.cf_retrieval import CFGraphRecaller


class SupportedCFGraphRecaller(CFGraphRecaller):
    """Keep all history seeds/neighbors; gate candidate columns before TopK.

    Counts use positive graph edges, not edge weight, repeated observations,
    query-time histories, validation clicks or test clicks. The caller binds the
    graph to its training cutoff. DSSM is independent and remains unrestricted.
    """

    def _initialize(self, graph, users, items, swing, config):
        threshold = config.get("min_item_users")
        if (isinstance(threshold, (bool, np.bool_))
                or not isinstance(threshold, (int, np.integer)) or threshold < 1):
            raise ValueError("min_item_users must be a positive integer")
        super()._initialize(graph, users, items, swing, config)
        observed = self.graph.copy()
        observed.sum_duplicates()
        observed.eliminate_zeros()
        counts = np.diff(observed.tocsc().indptr)
        self.item_user_counts = counts
        self.item_user_counts.setflags(write=False)
        self.eligible_items = counts >= int(threshold)
        self.eligible_items.setflags(write=False)
        # Similarities/IDF were computed on the original graph by the parent.
        # Only destinations are masked, so cold history can still find peers.
        self.graph = self.graph.multiply(self.eligible_items).tocsr()
        self.swing = self.swing.multiply(self.eligible_items).tocsr()
        self.graph.eliminate_zeros()
        self.swing.eliminate_zeros()
