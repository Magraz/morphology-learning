REGISTRY = {}

from .basic_controller import BasicMAC
REGISTRY["basic_mac"] = BasicMAC

from .dcg_controller import DeepCoordinationGraphMAC
REGISTRY["dcg_mac"] = DeepCoordinationGraphMAC

# NOTE: the alternative controllers (dcg_noshare_mac, cg_mac, low_rank_q) are
# vendored but not registered here: they depend on the compiled `torch_scatter`
# package (removed during integration) and are out of scope for the framework
# adapter. Re-add them here if/when they are ported.
