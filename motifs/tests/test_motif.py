from collections import Counter

from unittest import TestCase, skip

from motifs.motif import Motif, infer_event_type, InvalidEventTypeError


class TestMotifBuilding(TestCase):
    """ Testing the construction of motifs from various data types. """

    def test_(self):
        return None


class TestMotifFunction(TestCase):
    """ Testing the various built in functions of a motif. """

    def test_is_string(self):
        s = Motif.lettermap[0]
        self.assertTrue(s == 'A')


class TestMotifManipulation(TestCase):
    """ Testing that the motif class behaves in the correct manner when manipulated. """

    def setUp(self):
        self.motif_list = []
        self.motif_list.append(Motif([(0, 1, 1), (1, 2, 2)]))
        self.motif_list.append(Motif([(3, 4, 1), (4, 5, 2)]))
        self.motif_list.append(Motif([(3, 5, 1), (5, 3, 2)]))

    def tearDown(self):
        self.motif_list = None

    def test_correct_binning(self):
        count = dict(Counter(self.motif_list))
        self.assertDictEqual(count, {Motif('AB-BC'): 2, Motif('AB-BA'): 1})

    def test_string_equivalence(self):
        self.assertEqual(str(Motif('AB-BA')), 'AB-BA')
        self.assertEqual(str(Motif('ABr-BAr')), 'ABr-BAr')

    # self.assertEqual(Motif('A1B2-B2A1'), 'A1B2-B2A1')


class TestEventInference(TestCase):
    def test_plain_event(self):
        event = (0, 1, 2)
        self.assertListEqual(infer_event_type(event), ['source', 'target', 'time'])

    def test_duration_event(self):
        event = (0, 1, 2, 3)
        self.assertListEqual(infer_event_type(event), ['source', 'target', 'time', 'duration'])

    def test_edge_color_event(self):
        event = (0, 1, 2, 'r')
        self.assertListEqual(infer_event_type(event), ['source', 'target', 'time', 'edge_color'])

    def test_duration_and_edge_color_event(self):
        event = (0, 1, 2, 'r', 5)
        self.assertListEqual(infer_event_type(event), ['source', 'target', 'time', 'edge_color', 'duration'])
        event = (0, 1, 2, 5, 'r')
        self.assertListEqual(infer_event_type(event), ['source', 'target', 'time', 'duration', 'edge_color'])

    @skip("Skipping until working as expected!")
    def test_invalid_event(self):
        event = (0, 1)
        self.assertRaises(InvalidEventTypeError, infer_event_type(event))
        event = (1, 1, 1, 1, 1, 1, 1, 1, 1)
        self.assertRaises(InvalidEventTypeError, infer_event_type(event))
