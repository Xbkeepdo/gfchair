import unittest
import torch

from scripts.run_ffn_output_cosine import direction_projections
from features.ffn_visual_path_attribution import streaming_vector_path_statistics


class OutputCosineTest(unittest.TestCase):
    def test_vjp_matches_explicit_components_and_fixed_output_direction(self):
        torch.manual_seed(21)
        dtype = torch.float64
        weight = torch.randn(5,5,dtype=dtype)
        bias = torch.randn(5,dtype=dtype)
        fn = lambda x: torch.tanh(x @ weight + bias)
        z = torch.randn(3,5,dtype=dtype)
        writes = .1*torch.randn(7,3,5,dtype=dtype)
        output = fn(z).detach()
        projections,norms,clean = direction_projections(fn,z,writes,output)
        stats = streaming_vector_path_statistics(ffn_map=fn,z=z,writes=writes,
            integration_points=4,method='gauss_legendre',token_chunk_size=3,save_components=True)
        expected = torch.einsum('mtd,td->tm',stats.components,output/output.norm(dim=-1)[:,None])
        torch.testing.assert_close(projections[0],expected,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(projections[1],stats.path_signed_q,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(norms[1],stats.net_strength)
        # A bias makes the actual FFN output direction differ from its finite difference.
        self.assertGreater(float((projections[0]-projections[1]).abs().max()),1e-4)
        torch.testing.assert_close(clean,output)

    def test_zero_writes_and_zero_output_direction(self):
        z = torch.ones(2,4,dtype=torch.float64)
        writes = torch.zeros(3,2,4,dtype=torch.float64)
        p,_,_ = direction_projections(torch.sin,z,writes,torch.zeros_like(z))
        self.assertTrue(torch.equal(p,torch.zeros_like(p)))


if __name__ == '__main__': unittest.main()
