"""
Forward-pass smoke tests for ml_models/architectures/.
No training — just instantiate each model and run a single batch through it.
Input shape convention: (B, C, T) = (batch, channels, timesteps).
"""
import pytest
import torch
from ml_models.architectures import (
    EncoderTSC, Classifier_RESNET, Classifier_FCN, Classifier_MCDCNN,
    Classifier_TIMECNN, InceptionTime, TST, Classifier_LSTM, Classifier_GRU,
    LogisticClassifier,
    MLP_LSTM_Attention, VAE_RUL, CNN_LSTM_Regressor, AttentionLSTMRegressor,
    TSTRegressor, TCNRegressor, LSTM_FCN_Regressor, TFT_Regressor,
    LSTM_Regressor, BiLSTM_Regressor, LinearRegressor,
    NAMTS_Classifier, NAMTS_Regressor,
    SoftDecisionTreeClassifier, SoftDecisionTreeRegressor,
    AttnPoolClassifier, AttnPoolRegressor,
)

B, C, T = 4, 6, 50   # small batch for speed
NB_CLASSES = 3
INPUT_SHAPE = (T, C)  # models expect (T, C) as input_shape arg


# ---- Classifiers ----

def test_encoder_tsc():
    m = EncoderTSC(INPUT_SHAPE, NB_CLASSES)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B, NB_CLASSES)

def test_classifier_resnet():
    m = Classifier_RESNET(INPUT_SHAPE, NB_CLASSES)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B, NB_CLASSES)

def test_classifier_fcn():
    m = Classifier_FCN(INPUT_SHAPE, NB_CLASSES)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B, NB_CLASSES)

def test_classifier_mcdcnn():
    m = Classifier_MCDCNN(INPUT_SHAPE, NB_CLASSES)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B, NB_CLASSES)

def test_classifier_timecnn():
    m = Classifier_TIMECNN(INPUT_SHAPE, NB_CLASSES)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B, NB_CLASSES)

def test_inceptiontime():
    m = InceptionTime(INPUT_SHAPE, NB_CLASSES)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B, NB_CLASSES)

def test_tst_classifier():
    m = TST(INPUT_SHAPE, NB_CLASSES, d_model=32, n_heads=4, num_layers=2, d_ff=64, patch_len=8, stride=4)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B, NB_CLASSES)

def test_classifier_lstm():
    m = Classifier_LSTM(INPUT_SHAPE, NB_CLASSES, hidden_size=32, num_layers=2)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B, NB_CLASSES)

def test_classifier_gru():
    m = Classifier_GRU(INPUT_SHAPE, NB_CLASSES, hidden_size=32, num_layers=2)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B, NB_CLASSES)

def test_logistic_classifier():
    m = LogisticClassifier(INPUT_SHAPE, NB_CLASSES)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B, NB_CLASSES)


# ---- Regressors ----

def test_mlp_lstm_attention():
    m = MLP_LSTM_Attention(input_dim=C, output_dim=1, hidden_dim=16, lstm_hidden_dim=16)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B, 1)

def test_vae_rul():
    m = VAE_RUL(timesteps=T, input_dim=C, intermediate_dim=16, latent_dim=4)
    y, mu, logvar = m(torch.randn(B, C, T))
    assert y.shape == (B, 1)
    assert mu.shape == (B, 4)
    assert logvar.shape == (B, 4)

def test_cnn_lstm_regressor():
    m = CNN_LSTM_Regressor(INPUT_SHAPE, conv_channels=(16, 32), lstm_hidden=32)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B,)

def test_attention_lstm_regressor():
    m = AttentionLSTMRegressor(INPUT_SHAPE, hidden_size=32, num_layers=2)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B,)

def test_tst_regressor():
    m = TSTRegressor(INPUT_SHAPE, d_model=32, n_heads=4, num_layers=2, d_ff=64, patch_len=8, stride=4)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B,)

def test_tcn_regressor():
    m = TCNRegressor(INPUT_SHAPE, channels=(16, 32), kernel_size=3)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B,)

def test_lstm_fcn_regressor():
    m = LSTM_FCN_Regressor(INPUT_SHAPE, lstm_hidden=32, fcn_channels=(16, 32, 16))
    out = m(torch.randn(B, C, T))
    assert out.shape == (B,)

def test_tft_regressor():
    m = TFT_Regressor(INPUT_SHAPE, d_model=16, n_heads=2, n_attn_layers=1, d_ff=32, lstm_hidden=16)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B,)

def test_lstm_regressor():
    m = LSTM_Regressor(INPUT_SHAPE, hidden_size=32, num_layers=2)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B,)

def test_bilstm_regressor():
    m = BiLSTM_Regressor(INPUT_SHAPE, hidden_size=32, num_layers=2)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B,)

def test_linear_regressor():
    m = LinearRegressor(INPUT_SHAPE)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B,)


# ---- Interpretable ----

def test_namts_classifier():
    m = NAMTS_Classifier(INPUT_SHAPE, NB_CLASSES, d_hidden=16)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B, NB_CLASSES)

def test_namts_regressor():
    m = NAMTS_Regressor(INPUT_SHAPE, d_hidden=16)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B,)

def test_soft_dt_classifier():
    m = SoftDecisionTreeClassifier(INPUT_SHAPE, NB_CLASSES, depth=3)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B, NB_CLASSES)

def test_soft_dt_regressor():
    m = SoftDecisionTreeRegressor(INPUT_SHAPE, depth=3)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B,)

def test_attnpool_classifier():
    m = AttnPoolClassifier(INPUT_SHAPE, NB_CLASSES)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B, NB_CLASSES)

def test_attnpool_regressor():
    m = AttnPoolRegressor(INPUT_SHAPE)
    out = m(torch.randn(B, C, T))
    assert out.shape == (B,)
