use anyhow;
use candle_transformers::models::stable_diffusion::{self, vae::AutoEncoderKL};
use candle_core::{Device, DType};

use crate::stable_diffusion::stable_diffusion_files;

fn get_vae(vae_file: Option<String>, stable_diffusion_config: &stable_diffusion::StableDiffusionConfig, device: &Device, dtype: DType) -> anyhow::Result<AutoEncoderKL>{

    let vae_sd = stable_diffusion_files::StableDiffusionFiles::Vae;

    let vae_weights_file = vae_sd.get(vae_file, true)?;

    let vae = stable_diffusion_config.build_vae(vae_weights_file, &device, dtype)?;

    Ok(vae)

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_vae() {

        let width = Some(640 as usize);
        let height: Option<usize> = Some(480 as usize);
        let sd_config = stable_diffusion::StableDiffusionConfig::v1_5(None, height, width);

        let vae = get_vae(None, &sd_config, &Device::Cpu, DType::F16);
        assert!(vae.is_ok());

    }
}