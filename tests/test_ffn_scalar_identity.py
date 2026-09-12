import unittest
import numpy as np
import torch
from scripts.analyze_ffn_scalar_identity import token_metrics


class ScalarIdentityTest(unittest.TestCase):
    def test_resume_partitions_cover_remaining_images_once(self):
        from scripts.run_ffn_scalar_subspace import pending_images
        ids=[11,12,20,21,30,35];done={11,20}
        left=pending_images(ids,done,0);right=pending_images(ids,done,1)
        self.assertFalse(set(left)&set(right))
        self.assertEqual(set(left+right),set(ids)-done)
        self.assertEqual(pending_images(ids,set(ids),0),[])
        self.assertEqual(pending_images(ids,done),[12,21,30,35])

    def test_projected_identity_can_hide_leakage_and_rotation_is_not_scaling(self):
        from features.ffn_scalar_subspace import visual_basis,scalar_subspace
        q=torch.eye(4,dtype=torch.float64)[:,:2]
        y=2*q+torch.eye(4,dtype=torch.float64)[:,2:]
        s,b,_,_=scalar_subspace(q,y,reference_svd=True)
        np.testing.assert_allclose(s['scalar_error_projected'],0)
        np.testing.assert_allclose(s['scalar_error_full'],1/np.sqrt(5))
        np.testing.assert_allclose(s['leakage_ratio'],1/np.sqrt(5))
        s,_,_,_=scalar_subspace(q,torch.eye(4,dtype=torch.float64)[:,2:],reference_svd=True)
        np.testing.assert_allclose(s['scalar_error_full'],1)
        np.testing.assert_allclose(s['leakage_ratio'],1)
        self.assertTrue(np.isnan(s['scalar_error_projected']))
        self.assertTrue(np.isnan(s['sigma_B_p90_p10']))
        rotation=torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float64)
        s,_,_,_=scalar_subspace(q,q@rotation)
        np.testing.assert_allclose(s['sigma_B_cv'],0)
        np.testing.assert_allclose(s['scalar_error_full'],1)
        s,_,_,_=scalar_subspace(q,-3*q)
        np.testing.assert_allclose(s['scalar_error_full'],0)
        np.testing.assert_allclose(s['span_c'],-3)
        a=torch.column_stack([q[:,0],2*q[:,0],q[:,1],torch.zeros(4)])
        qb,audit=visual_basis(a)
        self.assertEqual(audit['rank'],2)
        torch.testing.assert_close(qb@qb.T,q@q.T)

    def test_basis_probes_do_not_change_integral_path(self):
        from tests.test_ffn_source_composition import RMS,Gated
        from scripts.extract_ffn_input_rotation import factored_components
        from features.ffn_visual_path_attribution import quadrature_rule
        torch.manual_seed(21)
        norm,ffn=RMS(1e-5).requires_grad_(False),Gated().requires_grad_(False)
        z=torch.randn(1,3,dtype=torch.float64)
        a=torch.randn(5,1,3,dtype=torch.float64)*.2
        q=torch.eye(3,dtype=torch.float64)[:,None,:]
        actual=factored_components(norm,ffn,z,a,directions=q)
        expected=torch.zeros_like(actual)
        for alpha,weight in zip(*[getattr(quadrature_rule('gauss_legendre',4,device=z.device,dtype=z.dtype),k) for k in ('nodes','weights')]):
            point=z-a.sum(0)+alpha*a.sum(0)
            for i,d in enumerate(q):
                expected[i]+=weight*torch.func.jvp(lambda x:ffn(norm(x)),(point,),(d,))[1]
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-11)

    def test_negative_scalar_rotation_and_nonuniform_gain(self):
        a=np.array([[1.,2.]])
        v=token_metrics(a,3*a,-np.ones_like(a))
        np.testing.assert_allclose(v['parallel_gain_mean'],-3)
        np.testing.assert_allclose(v['shared_scalar_error'],0)
        np.testing.assert_allclose(v['orthogonal_ratio_mean'],0)
        v=token_metrics(a,a,np.zeros_like(a))
        np.testing.assert_allclose(v['shared_scalar_error'],1)
        np.testing.assert_allclose(v['orthogonal_ratio_mean'],1)
        v=token_metrics(a,np.array([[1.,4.]]),np.ones_like(a))
        np.testing.assert_allclose(v['orthogonal_ratio_mean'],0)
        self.assertGreater(v['shared_scalar_error'][0],0)
        np.testing.assert_allclose(v['common_c'],1.8)


if __name__=='__main__':unittest.main()
