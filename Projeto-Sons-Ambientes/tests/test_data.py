import unittest

import numpy as np

from esc50.data import (
    fold_indices,
    label_maps,
    read_metadata,
    validate_fold_spec,
    validate_metadata,
)


class MetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = read_metadata()

    def test_official_metadata_is_balanced(self):
        validate_metadata(self.metadata)
        labels, label2id, id2label = label_maps(self.metadata)
        self.assertEqual(len(labels), 50)
        self.assertEqual(len(label2id), 50)
        self.assertEqual(len(id2label), 50)
        for fold in range(1, 6):
            counts = np.bincount(
                self.metadata.loc[self.metadata.fold == fold, "target"], minlength=50
            )
            np.testing.assert_array_equal(counts, np.full(50, 8))

    def test_screening_has_expected_examples_per_class(self):
        train = self.metadata[self.metadata.fold.isin([1, 2, 3])]
        validation = self.metadata[self.metadata.fold == 4]
        test = self.metadata[self.metadata.fold == 5]
        np.testing.assert_array_equal(np.bincount(train.target, minlength=50), np.full(50, 24))
        np.testing.assert_array_equal(
            np.bincount(validation.target, minlength=50), np.full(50, 8)
        )
        np.testing.assert_array_equal(np.bincount(test.target, minlength=50), np.full(50, 8))

    def test_fold_order_is_reproducible(self):
        first = fold_indices(self.metadata, [1, 2, 3], seed=42)
        second = fold_indices(self.metadata, [1, 2, 3], seed=42)
        different = fold_indices(self.metadata, [1, 2, 3], seed=43)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_fold_leakage_is_rejected(self):
        self.assertEqual(validate_fold_spec([1, 2, 3], 4, 5), (1, 2, 3))
        with self.assertRaises(ValueError):
            validate_fold_spec([1, 2, 5], 4, 5)
        with self.assertRaises(ValueError):
            validate_fold_spec([1, 2, 3], 4, 4)


if __name__ == "__main__":
    unittest.main()
