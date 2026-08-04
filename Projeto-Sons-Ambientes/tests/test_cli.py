import unittest

from esc50.cli import build_parser


class CLITests(unittest.TestCase):
    def test_train_contract(self):
        args = build_parser().parse_args(
            [
                "train",
                "--model",
                "ast",
                "--train-folds",
                "1,2,3",
                "--validation-fold",
                "4",
                "--test-fold",
                "5",
            ]
        )
        self.assertEqual(args.train_folds, (1, 2, 3))
        self.assertEqual(args.validation_fold, 4)
        self.assertEqual(args.test_fold, 5)

    def test_cross_validation_contract(self):
        args = build_parser().parse_args(
            [
                "cross-validate",
                "--models",
                "ast,whisper",
                "--model-epochs",
                "ast=3,whisper=5",
            ]
        )
        self.assertEqual(args.models, ("ast", "whisper"))
        self.assertEqual(args.model_epochs, {"ast": 3, "whisper": 5})


if __name__ == "__main__":
    unittest.main()
