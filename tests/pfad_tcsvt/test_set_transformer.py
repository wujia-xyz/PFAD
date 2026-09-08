"""Check source fidelity and the masking invariants used in the comparison."""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "src"))
from pfad_tcsvt.models import MaskedMAB, SetTransformerSelector


def test_mab_matches_upstream_on_unpadded_data():
    spec = importlib.util.spec_from_file_location("upstream_set_transformer",ROOT / "third_party/set_transformer/modules.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(7)
    upstream = module.MAB(8,8,32,4,ln=True)
    adapted = MaskedMAB(8,8,32,4)
    adapted.load_state_dict(upstream.state_dict())
    query,key = torch.randn(5,3,8),torch.randn(5,4,8)
    torch.testing.assert_close(upstream(query,key),adapted(query,key,torch.ones(5,4)),rtol=0,atol=0)


def test_classifier_ignores_padding_and_record_order():
    torch.manual_seed(8)
    model = SetTransformerSelector(32,4).eval()
    source = torch.randn(4,5,8)
    mask = torch.tensor([[1,0,0,0,0],[1,1,0,0,0],[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.float)
    context = torch.randn(4,19)
    with torch.no_grad():
        expected = model(source,mask,context)
        changed = source.clone()
        changed[~mask.bool()] = torch.randn_like(changed[~mask.bool()])*100
        torch.testing.assert_close(model(changed,mask,context),expected,rtol=1e-5,atol=1e-6)
        order = torch.tensor([3,0,4,1,2])
        torch.testing.assert_close(model(source[:,order],mask[:,order],context),expected,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(model(source[:1,:1],mask[:1,:1],context[:1]),expected[:1],rtol=1e-5,atol=1e-6)
