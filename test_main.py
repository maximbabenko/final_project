import main

def test():
    assert main.for_test == 'Success'

def test_model():
    assert main.model_ckpt == "papluca/xlm-roberta-base-language-detection"
