import os
import torchaudio

def load_align_model(language_code, device, align_library="torchaudio"):
    if align_library != "torchaudio":
        raise ValueError("Only 'torchaudio' align_library is supported in this patch.")
    
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
    model = bundle.get_model().to(device)
    metadata = bundle._params

    return model, metadata
