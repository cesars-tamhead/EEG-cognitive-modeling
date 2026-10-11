import unittest,numpy as np,pandas as pd
from run_tuning import trimmed_features,build

class Checks(unittest.TestCase):
 def test_trimmed_feature_reduces_outlier_without_dropping_rows(self):
  d=pd.DataFrame({'subject_id':['s']*10,'group':['control']*10,'condition':['S1 obj']*10,'trial':range(10),'logcov_0':[1.]*9+[101.]});r=trimmed_features(d);self.assertAlmostEqual(r['S1 obj__logcov_0'].iloc[0],1);self.assertEqual(len(d),10)
 def test_constant_features_preserved_by_all_aggregations(self):
  rows=[{'subject_id':s,'group':g,'condition':c,'trial':t,'logcov_0':2.,'logcorr_0':3.} for s,g in [('s1','control'),('s2','alcoholic')] for c in ['S1 obj','S2 match','S2 nomatch'] for t in range(10)];tables,specs,families=build(pd.DataFrame(rows));self.assertEqual(len(specs),18);self.assertEqual(len(families['original_spatial']),3)
  for name,f in tables.items():self.assertEqual(len(f),2);np.testing.assert_allclose(f.iloc[:,2:].to_numpy(),2 if name.startswith('cov_') else 3)
 def test_original_tie_order_is_preserved(self):
  d=pd.DataFrame([{'subject_id':'s','group':'control','condition':'S1 obj','trial':0,'logcov_0':1.,'logcorr_0':1.}]);_,s,_=build(d);self.assertEqual([x[0] for x in s[:3]],['cov_mean_pca6_C0.1','cov_median_pca6_C0.1','corr_mean_pca6_C0.1'])
if __name__=='__main__':unittest.main()
