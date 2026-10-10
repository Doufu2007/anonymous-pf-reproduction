"""Hand-computed gold and leakage-boundary tests for real-data intake."""
import unittest
import numpy as np
from data import barcode_to_subject, control_segments, interpolate_truth, select_batches, filter_inputs

class IntakeTests(unittest.TestCase):
    def test_barcode_is_not_subject_id(self):
        self.assertEqual(barcode_to_subject(np.array([[1,5],[6,72],[11,18]]))[18],11)

    def test_piecewise_controls_are_causal(self):
        odom=np.array([[0.,1.,0.],[1.,2.,.2],[2.,3.,.5]])
        np.testing.assert_allclose(control_segments(odom,.5,1.5),[[.5,1.,0.],[.5,2.,.2]])

    def test_truth_wrap_interpolation(self):
        gt=np.array([[0.,0.,0.,3.],[2.,2.,4.,-3.]])
        value,valid=interpolate_truth(gt,np.array([1.]))
        np.testing.assert_allclose(value[0,:2],[1.,2.])
        self.assertAlmostEqual(abs(value[0,2]),np.pi)
        self.assertTrue(valid[0])

    def test_batch_does_not_merge_different_times(self):
        meas=np.array([[1.,6,1.,0.],[1.,7,2.,.1],[1.2,6,1.,0.],[2.1,6,1.,0.]])
        batches=select_batches(meas,0.,5.,1.)
        self.assertEqual([x[0] for x in batches],[1.,2.1])
        self.assertEqual(batches[0][1].shape,(2,4))

    def test_truth_cannot_enter_filter_inputs(self):
        fake={'odometry_segments':[], 'event_times':np.array([]),'dt':np.array([]),
              'observations':[], 'landmark_xy':[], 'known_map':np.zeros((2,2)),
              'state_min':np.zeros(3),'state_max':np.ones(3),'start_time':0.,
              'truth':np.full((5,3),99999.),'truth_valid':np.ones(5,dtype=bool)}
        self.assertNotIn('truth',filter_inputs(fake))
        self.assertNotIn('truth_valid',filter_inputs(fake))

if __name__=='__main__': unittest.main()
