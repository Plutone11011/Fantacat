use std;
use candle_transformers::models::based::Model;
use hf_hub::api::sync::Api;
use anyhow::Result;

use crate::stable_diffusion::constants;

#[derive(Debug)]
pub enum StableDiffusionFiles{

    Tokenizer,
    Clip,
    Unet,
    Vae
}

pub trait ModelFileBuild {
    fn get_repo(&self, sd_file: &StableDiffusionFiles) -> &str{
        self.get_repo_with_precision(sd_file, None)
    }
    fn get_repo_with_precision(&self, sd_file: &StableDiffusionFiles, use_f16: Option<bool>) -> &str;
    fn get_tokenizer_filepath(&self) -> &str {
        constants::MODELFILE_TOKENIZER
    }

    fn get_clip_filepath(&self, use_f16: bool) -> &str {
        if use_f16 { constants::MODELFILE_CLIP_FP16 } else { constants::MODELFILE_CLIP }
    }

    fn get_unet_filepath(&self, use_f16: bool) -> &str {
        if use_f16 { constants::MODELFILE_UNET_FP16 } else { constants::MODELFILE_UNET }
    }

    fn get_vae_filepath(&self, use_f16: bool) -> &str {
        if use_f16 { constants::MODELFILE_VAE_FP16 } else { constants::MODELFILE_VAE }
    }
    fn get(&self, sd_file: &StableDiffusionFiles, filename: Option<String>, use_f16: bool) -> Result<std::path::PathBuf> {
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let repo = self.get_repo_with_precision(sd_file, Some(use_f16));
                let filepath = match sd_file {
                    StableDiffusionFiles::Tokenizer => self.get_tokenizer_filepath(),
                    StableDiffusionFiles::Clip => {
                        self.get_clip_filepath(use_f16)
                    },
                    StableDiffusionFiles::Unet => {
                        self.get_unet_filepath(use_f16)
                    },
                    StableDiffusionFiles::Vae => {
                        self.get_vae_filepath(use_f16)
                    }
                };

                let filename = Api::new()?.model(repo.to_string()).get(filepath)?;
                Ok(filename)
            }
        }
    }
}

pub struct StableDiffusion1_5 {}


impl ModelFileBuild for StableDiffusion1_5 {
    fn get_repo_with_precision(&self, sd_file: &StableDiffusionFiles, _use_f16: Option<bool>) -> &str {
        match sd_file {
            StableDiffusionFiles::Tokenizer => constants::REPO_TOKENIZER,
            StableDiffusionFiles::Clip|StableDiffusionFiles::Unet|StableDiffusionFiles::Vae => constants::REPO_1_5
        }
    }

    
}

pub struct StableDiffusion2_1{}

impl ModelFileBuild for StableDiffusion2_1 {
    fn get_repo_with_precision(&self, sd_file: &StableDiffusionFiles, _use_f16: Option<bool>) -> &str {
        match sd_file {
            StableDiffusionFiles::Tokenizer => constants::REPO_TOKENIZER,
            StableDiffusionFiles::Clip|StableDiffusionFiles::Unet|StableDiffusionFiles::Vae => constants::REPO_2_1
        }
    }
}


pub struct StableDiffusionTurbo{}

impl ModelFileBuild for StableDiffusionTurbo{
    fn get_repo_with_precision(&self, sd_file: &StableDiffusionFiles, _use_f16: Option<bool>) -> &str {
        match sd_file {
            StableDiffusionFiles::Tokenizer => constants::REPO_TOKENIZER_X1TURBO,
            StableDiffusionFiles::Clip|StableDiffusionFiles::Unet => constants::REPO_TURBO,
            StableDiffusionFiles::Vae => if _use_f16.unwrap_or(false) {constants::REPO_VAE_X1TURBO_FP16} else {constants::REPO_TURBO}
        }
    }

    fn get_vae_filepath(&self, use_f16: bool) -> &str {
        if use_f16 { constants::MODELFILE_VAE_X1TURBO_FP16 } else { constants::MODELFILE_VAE }
    }
}

pub struct StableDiffusionX1{}

impl ModelFileBuild for StableDiffusionX1{
    fn get_repo_with_precision(&self, sd_file: &StableDiffusionFiles, _use_f16: Option<bool>) -> &str {
        match sd_file {
            StableDiffusionFiles::Tokenizer => constants::REPO_TOKENIZER_X1TURBO,
            StableDiffusionFiles::Clip|StableDiffusionFiles::Unet => constants::REPO_X1,
            StableDiffusionFiles::Vae => if _use_f16.unwrap_or(false) {constants::REPO_VAE_X1TURBO_FP16} else {constants::REPO_TURBO}
        }
    }

    fn get_vae_filepath(&self, use_f16: bool) -> &str {
        if use_f16 { constants::MODELFILE_VAE_X1TURBO_FP16 } else { constants::MODELFILE_VAE }
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
        let unet_path = sd_version.get_unet_filepath(true);
        
        assert_eq!(unet_repo, "stable-diffusion-v1-5/stable-diffusion-v1-5");
        assert_eq!(unet_path, "unet/diffusion_pytorch_model.fp16.safetensors");
    }

    #[test]
    fn sd_files_vae() {
        let model_file = StableDiffusionFiles::Vae;
        let sd_version = StableDiffusion1_5{};
        let vae_repo = sd_version.get_repo(&model_file);
        let vae_path = sd_version.get_vae_filepath(false);
        
        assert_eq!(vae_repo, "stable-diffusion-v1-5/stable-diffusion-v1-5");
        assert_eq!(vae_path, "vae/diffusion_pytorch_model.safetensors");
    }
    #[test]
    fn sd_files_tokenizer() {
        let model_file = StableDiffusionFiles::Tokenizer;
        let sd_version = StableDiffusion1_5{};
        let tokenizer_repo = sd_version.get_repo(&model_file);
        let tokenizer_path = sd_version.get_tokenizer_filepath();
        
        assert_eq!(tokenizer_repo, "openai/clip-vit-base-patch32");
        assert_eq!(tokenizer_path, "tokenizer.json");
    }
    #[test]
    fn sd_files_encoder() {
        let model_file = StableDiffusionFiles::Clip;
        let sd_version = StableDiffusion1_5{};
        let encoder_repo = sd_version.get_repo(&model_file);
        let encoder_path = sd_version.get_clip_filepath(true);
        
        assert_eq!(encoder_repo, "stable-diffusion-v1-5/stable-diffusion-v1-5");
        assert_eq!(encoder_path, "text_encoder/model.fp16.safetensors");
    }
}