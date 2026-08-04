import unittest

import torch
from transformers import (
    ASTConfig,
    ASTForAudioClassification,
    Wav2Vec2Config,
    Wav2Vec2ForSequenceClassification,
    WhisperConfig,
    WhisperForAudioClassification,
)

from esc50.models import Esc50CNNConfig, Esc50CNNForAudioClassification


class ArchitectureSmokeTests(unittest.TestCase):
    def assert_valid_output(self, model, output):
        self.assertEqual(tuple(output.logits.shape), (2, 50))
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue(torch.isfinite(output.loss))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()

    def test_ast_forward(self):
        config = ASTConfig(
            num_mel_bins=16,
            max_length=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            frequency_stride=4,
            time_stride=4,
            num_labels=50,
        )
        model = ASTForAudioClassification(config)
        output = model(input_values=torch.randn(2, 32, 16), labels=torch.tensor([0, 1]))
        self.assert_valid_output(model, output)

    def test_wav2vec2_forward(self):
        config = Wav2Vec2Config(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            conv_dim=(8, 8, 8),
            conv_stride=(5, 2, 2),
            conv_kernel=(10, 3, 3),
            num_conv_pos_embeddings=16,
            num_conv_pos_embedding_groups=2,
            num_labels=50,
        )
        model = Wav2Vec2ForSequenceClassification(config)
        output = model(input_values=torch.randn(2, 800), labels=torch.tensor([0, 1]))
        self.assert_valid_output(model, output)

    def test_whisper_encoder_forward(self):
        config = WhisperConfig(
            num_mel_bins=16,
            d_model=16,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_layers=1,
            decoder_attention_heads=2,
            decoder_ffn_dim=32,
            max_source_positions=50,
            classifier_proj_size=8,
            num_labels=50,
        )
        model = WhisperForAudioClassification(config)
        output = model(input_features=torch.randn(2, 16, 100), labels=torch.tensor([0, 1]))
        self.assert_valid_output(model, output)

    def test_cnn_forward(self):
        model = Esc50CNNForAudioClassification(
            Esc50CNNConfig(num_labels=50, channels=(4, 8))
        )
        output = model(input_features=torch.randn(2, 1, 32, 64), labels=torch.tensor([0, 1]))
        self.assert_valid_output(model, output)


if __name__ == "__main__":
    unittest.main()
