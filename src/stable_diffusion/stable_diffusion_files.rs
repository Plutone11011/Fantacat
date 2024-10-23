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

impl StableDiffusionFiles{

    pub fn get_repo(&self) -> &str{


        match self {
            Self::Tokenizer => "openai/clip-vit-base-patch32",
            Self::Clip|Self::Unet|Self::Vae => "stable-diffusion-v1-5/stable-diffusion-v1-5"
        }
    }

    pub fn get_path(&self, use_f16: bool) -> &str{

        match self {
            Self::Tokenizer => "tokenizer.json",
            Self::Clip => {
                if use_f16 {
                    "text_encoder/model.fp16.safetensors"
                }
                else {
                    "text_encoder/model.safetensors"
                }
            },
            Self::Unet => {
                if use_f16 {
                    "unet/diffusion_pytorch_model.fp16.safetensors"
                }
                else {
                    "unet/diffusion_pytorch_model.safetensors"
                }
            },
            Self::Vae => {
                if use_f16 {
                    "vae/diffusion_pytorch_model.fp16.safetensors"
                }
                else {
                    "vae/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    pub fn get(&self, filename: Option<String>, use_f16: bool) -> Result<std::path::PathBuf>{
        // Returns filename if it exists
        // 
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let repo = self.get_repo();
                let filepath = self.get_path(use_f16);

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
    fn test_unet() {
        let model_file = StableDiffusionFiles::Unet;
        let unet_repo = model_file.get_repo();
        let unet_path = model_file.get_path(true);
        
        assert_eq!(unet_repo, "stable-diffusion-v1-5/stable-diffusion-v1-5");
        assert_eq!(unet_path, "unet/diffusion_pytorch_model.fp16.safetensors");
    }

    #[test]
    fn test_vae() {
        let model_file = StableDiffusionFiles::Vae;
        let vae_repo = model_file.get_repo();
        let vae_path = model_file.get_path(false);
        
        assert_eq!(vae_repo, "stable-diffusion-v1-5/stable-diffusion-v1-5");
        assert_eq!(vae_path, "vae/diffusion_pytorch_model.safetensors");
    }
    #[test]
    fn test_tokenizer() {
        let model_file = StableDiffusionFiles::Tokenizer;
        let tokenizer_repo = model_file.get_repo();
        let tokenizer_path = model_file.get_path(true);
        
        assert_eq!(tokenizer_repo, "openai/clip-vit-base-patch32");
        assert_eq!(tokenizer_path, "tokenizer.json");
    }
    #[test]
    fn test_encoder() {
        let model_file = StableDiffusionFiles::Clip;
        let encoder_repo = model_file.get_repo();
        let encoder_path = model_file.get_path(true);
        
        assert_eq!(encoder_repo, "stable-diffusion-v1-5/stable-diffusion-v1-5");
        assert_eq!(encoder_path, "text_encoder/model.fp16.safetensors");
    }
}