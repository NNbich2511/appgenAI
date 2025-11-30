from genAI.cliptexttrans import CLIP
from genAI.VAE import VAE_Encoder
from genAI.convert import load_from_standard_weights
from genAI.decoder import VAE_Decoder
from genAI.Unetmodel import Diffusion


def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)
    print(state_dict)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }


from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="Envvi/Inkpunk-Diffusion",
    filename="Inkpunk-Diffusion-v2.ckpt"
)

#state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

print(preload_models_from_standard_weights("D://v1-5-pruned-emaonly.ckpt", device= "cpu"))