import unittest
import numpy as np
from scripts.analyze_ffn_input_geometry import geometry


class GeometryTest(unittest.TestCase):
    def test_factored_integral_matches_direct_unfrozen_norm_jvp(self):
        import torch
        from tests.test_ffn_source_composition import RMS, Gated
        from scripts.extract_ffn_input_rotation import factored_components
        from features.ffn_visual_path_attribution import streaming_vector_path_statistics
        torch.manual_seed(19)
        norm=RMS(1e-5).requires_grad_(False)
        ffn=Gated().requires_grad_(False)
        z=torch.randn(2,3,dtype=torch.float64)
        a=torch.randn(7,2,3,dtype=torch.float64)*.2
        fn=lambda x:ffn(norm(x))
        reference=streaming_vector_path_statistics(ffn_map=fn,z=z,writes=a,method='gauss_legendre',
            integration_points=4,token_chunk_size=3,save_components=True)
        with torch.no_grad(): actual=factored_components(norm,ffn,z,a,chunk=3)
        torch.testing.assert_close(actual,reference.components,rtol=1e-11,atol=1e-12)

    def test_rotation_is_not_endpoint_cosine(self):
        import torch
        from scripts.extract_ffn_input_rotation import rotation
        a=torch.tensor([[[1.,0.]],[[0.,1.]],[[0.,0.]]])
        e=torch.tensor([[[0.,2.]],[[0.,-3.]],[[0.,0.]]])
        cos,missing=rotation(a,e)
        np.testing.assert_allclose(cos[0,:2], [0.,-1.])
        self.assertTrue(np.isnan(float(cos[0,2])))
        self.assertEqual(int(missing[0]),1)
        cos,missing=rotation(torch.zeros_like(a),torch.zeros_like(a))
        self.assertTrue(torch.isnan(cos).all())
        self.assertEqual(int(missing[0]),3)

    def test_gain_weights_and_exact_vector_cancellation(self):
        # a1=(1,0), a2=(3,0), e1=(2,0), e2=(-3,0):
        # mean gain 1.5, weighted gain 1.25, cancellation 1/5.
        values, counts = geometry([[1,3,0]], [[2,3,0]], [1], [0])
        self.assertEqual(values['amp_mean'][0], 1.5)
        self.assertEqual(values['gross_gain'][0], 1.25)
        self.assertEqual(values['cancellation'][0], .2)
        self.assertEqual(values['amp_fraction_gt1'][0], .5)
        self.assertEqual(counts['zero_write'], 1)
        values, _ = geometry([[0,0]], [[0,0]], [0], [0])
        self.assertTrue(np.isnan(values['cancellation'][0]))
        self.assertTrue(np.isnan(values['amp_mean'][0]))
        with self.assertRaises(ValueError): geometry([[1]], [[1]], [2], [0])


if __name__ == '__main__': unittest.main()
