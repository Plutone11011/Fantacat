


#[derive(Debug)]
pub enum ModelFile{

    Tokenizer,
    Clip,
    Unet,
    Vae
}

impl ModelFile{

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
                    "text-encoder/model.fp16.safetensors"
                }
                else {
                    "text-encoder/model.safetensors"
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
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_repo() {
        let model_file = ModelFile::Unet;
        let unet_repo = model_file.get_repo();
        
        assert_eq!(unet_repo, "stable-diffusion-v1-5/stable-diffusion-v1-5");
    }
}