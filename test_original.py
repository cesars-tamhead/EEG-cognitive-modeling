"""Validation tests; synthetic arrays test mathematics, never model accuracy."""
import unittest,numpy as np,pandas as pd
from pathlib import Path
from improve_original import spatial_epoch,log_vector,tables_and_specs,temporal_features

class Checks(unittest.TestCase):
 def test_covariance_finite_for_flat_electrode(self):
  x=np.random.default_rng(1).normal(size=(5,256));x[0]=0
  f,s=spatial_epoch(x);self.assertTrue(np.isfinite(f).all());self.assertTrue(0<=s<=1);self.assertEqual(len(f),30)
 def test_all_flat_is_finite(self):
  f,_=spatial_epoch(np.zeros((4,256)));self.assertTrue(np.isfinite(f).all())
 def test_covariance_invariant_to_dc_offset(self):
  x=np.random.default_rng(2).normal(size=(4,256));a,_=spatial_epoch(x);b,_=spatial_epoch(x+np.arange(4)[:,None]*10);np.testing.assert_allclose(a,b,atol=1e-10)
 def test_correlation_invariant_to_global_amplitude(self):
  x=np.random.default_rng(2).normal(size=(4,256));a,_=spatial_epoch(x);b,_=spatial_epoch(x*3);np.testing.assert_allclose(a[len(a)//2:],b[len(b)//2:],atol=1e-10)
 def test_logvector_preserves_symmetric_frobenius_norm(self):
  a=np.array([[2.,.2],[.2,1.]])
  e,q=np.linalg.eigh(a);l=(q*np.log(e))@q.T;self.assertAlmostEqual(np.linalg.norm(log_vector(a)),np.linalg.norm(l))
 def test_trial_identity_preserved(self):
  root=Path(__file__).resolve().parent;bp=root.parent/'eeg_validation_results/channel_features.csv';ep=root.parent/'eeg_v2_results/extended_channel_features.csv';spatial=root/'results/spatial_trials.csv'
  if not all(p.exists() for p in [bp,ep,spatial]):
   bp=root/'results/v2_features/channel_features.csv';ep=root/'results/v2_features/extended_channel_features.csv'
  if not all(p.exists() for p in [bp,ep,spatial]):self.skipTest('Run feature extraction first for the actual-data identity check')
  base=pd.read_csv(bp);ext=pd.read_csv(ep);sp=pd.read_csv(spatial);tables,specs=tables_and_specs(base,ext,sp)
  for task,versions in tables.items():
   n=20 if task=='subject' else 600
   for name,f in versions.items():self.assertEqual(len(f),n);self.assertEqual(f.subject_id.nunique(),20)
 def test_temporal_contrast_constant_bins_zero(self):
  d=pd.DataFrame({'subject_id':['s'],'group':['control'],'condition':['S1 obj'],'trial':[0],'channel':['FZ'],**{f'bin{i}_mean_uv':[4.] for i in range(8)}})
  t=temporal_features(d);self.assertTrue((t.iloc[0,4:].astype(float)==0).all())
if __name__=='__main__':unittest.main()
