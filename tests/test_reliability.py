import unittest
from pathlib import Path
import numpy as np
from src.reliability import fuse_reliability
from src.robustness_research import synthetic_cases,run_study,validate

class ReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame=synthetic_cases(300)
        cls.metrics,cls.pred,cls.meta=run_study(cls.frame)
    def test_no_evidence_abstains(self):
        self.assertIsNone(fuse_reliability({})['final_label'])
        self.assertIsNone(fuse_reliability({'video':{'score_norm':.99}},reliability={'video':0})['final_label'])
    def test_quality_downweights_artifact(self):
        result=fuse_reliability({'image':{'score_norm':.1},'video':{'score_norm':1}},reliability={'image':1,'video':.01})
        self.assertLess(result['final_score'],.12)
        self.assertEqual(result['final_label'],0)
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):fuse_reliability({'video':{'score_norm':1.1}})
        with self.assertRaises(ValueError):fuse_reliability({'video':{'score_norm':.1}},reliability={'video':-1})
    def test_group_leak_rejected(self):
        frame=self.frame.copy();frame.loc[200,'group_id']=frame.loc[0,'group_id']
        with self.assertRaises(ValueError):validate(frame)
    def test_test_labels_cannot_change_weights_thresholds_or_scores(self):
        frame=self.frame.copy();mask=frame.split.eq('test');frame.loc[mask,'label']=1-frame.loc[mask,'label']
        _,pred,meta=run_study(frame)
        self.assertEqual(meta['priors'],self.meta['priors']);self.assertEqual(meta['thresholds'],self.meta['thresholds'])
        np.testing.assert_allclose(pred.score,self.pred.score)
    def test_missing_all_is_not_normal(self):
        frame=self.frame.copy()
        for m in ['tabular','timeseries','image','video']:frame.loc[200,'score_'+m]=np.nan
        _,pred,_=run_study(frame)
        self.assertTrue(pred.loc[pred.case_id.eq('case-200'),'score'].isna().all())
    def test_reproducible_and_complete(self):
        metrics,pred,meta=run_study(self.frame)
        self.assertTrue(metrics.equals(self.metrics));self.assertEqual(len(metrics),30)
        self.assertTrue(metrics.coverage.eq(1).all())
    def test_research_dashboard(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/research.py'),default_timeout=90).run()
        self.assertEqual(len(app.exception),0)

    def test_detector_dashboard_reliable_and_abstaining_paths(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/app.py'),default_timeout=60).run()
        for checkbox in app.checkbox:
            if checkbox.label.startswith('Use ') and checkbox.label not in ['Use tabular modality','Use reliability-weighted fusion']:
                checkbox.set_value(False)
        app.run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any(m.label=='Final Fused Score' for m in app.metric))
        for slider in app.slider:
            if slider.label=='Tabular reliability':slider.set_value(0.)
        app.button[0].click().run()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any('Abstain' in w.value for w in app.warning))
