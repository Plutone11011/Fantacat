use std;
use hf_hub::api::sync::Api;
use anyhow::Result;

#[derive(Debug)]
pub enum StableDiffusionFiles{

    Tokenizer,
    Clip,
    Unet,
    Vae
}

pub trait ModelFileBuild {
    fn get_repo(&self, sd_file: &StableDiffusionFiles) -> &str;
    fn get_path(&self, sd_file: &StableDiffusionFiles, use_f16: bool) -> &str;
    fn get(&self, sd_file: &StableDiffusionFiles, filename: Option<String>, use_f16: bool) -> Result<std::path::PathBuf>;
}

pub struct StableDiffusion1_5 {}

impl StableDiffusion1_5 {
    const REPO_TOKENIZER: &str = "openai/clip-vit-base-patch32";
    const REPO_CLIP: &str = "stable-diffusion-v1-5/stable-diffusion-v1-5";
    const REPO_UNET: &str = "stable-diffusion-v1-5/stable-diffusion-v1-5";
    const REPO_VAE: &str = "stable-diffusion-v1-5/stable-diffusion-v1-5";

    const MODELFILE_TOKENIZER: &str = "tokenizer.json";
    const MODELFILE_CLIP: &str = "text_encoder/model.safetensors";
    const MODELFILE_CLIP_FP16: &str = "text_encoder/model.fp16.safetensors";
    const MODELFILE_UNET: &str = "unet/diffusion_pytorch_model.safetensors";
    const MODELFILE_UNET_FP16: &str = "unet/diffusion_pytorch_model.fp16.safetensors";
    const MODELFILE_VAE: &str = "vae/diffusion_pytorch_model.safetensors";
    const MODELFILE_VAE_FP16: &str = "vae/diffusion_pytorch_model.fp16.safetensors";

}

impl ModelFileBuild for StableDiffusion1_5 {
    fn get_repo(&self, sd_file: &StableDiffusionFiles) -> &str {
        match sd_file {
            StableDiffusionFiles::Tokenizer => StableDiffusion1_5::REPO_TOKENIZER,
            StableDiffusionFiles::Clip|StableDiffusionFiles::Unet|StableDiffusionFiles::Vae => StableDiffusion1_5::REPO_CLIP
        }
    }

    fn get_path(&self, sd_file: &StableDiffusionFiles, use_f16: bool) -> &str {
        match sd_file {
            StableDiffusionFiles::Tokenizer => StableDiffusion1_5::MODELFILE_TOKENIZER,
            StableDiffusionFiles::Clip => {
                if use_f16 {
                    StableDiffusion1_5::MODELFILE_CLIP_FP16
                }
                else {
                    StableDiffusion1_5::MODELFILE_CLIP
                }
            },
            StableDiffusionFiles::Unet => {
                if use_f16 {
                    StableDiffusion1_5::MODELFILE_UNET_FP16
                }
                else {
                    StableDiffusion1_5::MODELFILE_UNET
                }
            },
            StableDiffusionFiles::Vae => {
                if use_f16 {
                    StableDiffusion1_5::MODELFILE_VAE_FP16
                }
                else {
                    StableDiffusion1_5::MODELFILE_VAE
                }
            }
        }
    }

    fn get(&self, sd_file: &StableDiffusionFiles, filename: Option<String>, use_f16: bool) -> Result<std::path::PathBuf> {
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let repo = self.get_repo(sd_file);
                let filepath = self.get_path(sd_file, use_f16);

                let filename = Api::new()?.model(repo.to_string()).get(filepath)?;
                Ok(filename)
            }
        }
    }
}




#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sd_files_unet() {
        let model_file = StableDiffusionFiles::Unet;
        let sd_version = StableDiffusion1_5{};
        let unet_repo = sd_version.get_repo(&model_file);
        let unet_path = sd_version.get_path(&model_file, true);
        
        assert_eq!(unet_repo, "stable-diffusion-v1-5/stable-diffusion-v1-5");
        assert_eq!(unet_path, "unet/diffusion_pytorch_model.fp16.safetensors");
    }

    #[test]
    fn sd_files_vae() {
        let model_file = StableDiffusionFiles::Vae;
        let sd_version = StableDiffusion1_5{};
        let vae_repo = sd_version.get_repo(&model_file);
        let vae_path = sd_version.get_path(&model_file, false);
        
        assert_eq!(vae_repo, "stable-diffusion-v1-5/stable-diffusion-v1-5");
        assert_eq!(vae_path, "vae/diffusion_pytorch_model.safetensors");
    }
    #[test]
    fn sd_files_tokenizer() {
        let model_file = StableDiffusionFiles::Tokenizer;
        let sd_version = StableDiffusion1_5{};
        let tokenizer_repo = sd_version.get_repo(&model_file);
        let tokenizer_path = sd_version.get_path(&model_file,true);
        
        assert_eq!(tokenizer_repo, "openai/clip-vit-base-patch32");
        assert_eq!(tokenizer_path, "tokenizer.json");
    }
    #[test]
    fn sd_files_encoder() {
        let model_file = StableDiffusionFiles::Clip;
        let sd_version = StableDiffusion1_5{};
        let encoder_repo = sd_version.get_repo(&model_file);
        let encoder_path = sd_version.get_path(&model_file, true);
        
        assert_eq!(encoder_repo, "stable-diffusion-v1-5/stable-diffusion-v1-5");
        assert_eq!(encoder_path, "text_encoder/model.fp16.safetensors");
    }
}