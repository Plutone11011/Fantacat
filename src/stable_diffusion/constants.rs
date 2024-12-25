

pub const REPO_TOKENIZER: &str = "openai/clip-vit-base-patch32";
pub const REPO_TOKENIZER_X1TURBO: &str = "openai/clip-vit-large-patch14";
pub const REPO_TOKENIZER2: &str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k";
pub const REPO_1_5: &str = "stable-diffusion-v1-5/stable-diffusion-v1-5";
pub const REPO_2_1: &str = "stabilityai/stable-diffusion-2-1";
pub const REPO_X1: &str = "stabilityai/stable-diffusion-xl-base-1.0";
pub const REPO_TURBO: &str = "stabilityai/sdxl-turbo";
pub const REPO_VAE_X1TURBO_FP16: &str = "madebyollin/sdxl-vae-fp16-fix";

pub const MODELFILE_TOKENIZER: &str = "tokenizer.json";
pub const MODELFILE_CLIP: &str = "text_encoder/model.safetensors";
pub const MODELFILE_CLIP_FP16: &str = "text_encoder/model.fp16.safetensors";
pub const MODELFILE_UNET: &str = "unet/diffusion_pytorch_model.safetensors";
pub const MODELFILE_UNET_FP16: &str = "unet/diffusion_pytorch_model.fp16.safetensors";
pub const MODELFILE_VAE: &str = "vae/diffusion_pytorch_model.safetensors";
pub const MODELFILE_VAE_FP16: &str = "vae/diffusion_pytorch_model.fp16.safetensors";
pub const MODELFILE_VAE_X1TURBO_FP16: &str = "diffusion_pytorch_model.safetensors";
