use anyhow;
use candle_transformers::models::stable_diffusion::{self, vae::AutoEncoderKL};
use candle_core::{Device, DType};

use crate::stable_diffusion::stable_diffusion_files;

pub fn get_vae(vae_file: Option<String>, sd_version: &stable_diffusion_files::StableDiffusionVersion, stable_diffusion_config: &stable_diffusion::StableDiffusionConfig, device: &Device, dtype: DType) -> anyhow::Result<AutoEncoderKL>{

    let vae = stable_diffusion_files::StableDiffusionFiles::Vae;
    let sd = stable_diffusion_files::create_sd_from_version(sd_version);
    let vae_weights_file = sd.get(&vae, vae_file, true)?;

    let vae = stable_diffusion_config.build_vae(vae_weights_file, &device, dtype)?;

    Ok(vae)

}

pub fn get_vae_scale(sd_version: &stable_diffusion_files::StableDiffusionVersion) -> f64{
    match sd_version {
        stable_diffusion_files::StableDiffusionVersion::V1_5
         | stable_diffusion_files::StableDiffusionVersion::V2_1
         | stable_diffusion_files::StableDiffusionVersion::Xl => 0.18215,
         stable_diffusion_files::StableDiffusionVersion::Turbo => 0.13025,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vae() {

        let width = Some(640 as usize);
        let height: Option<usize> = Some(480 as usize);
        let sd_config = stable_diffusion::StableDiffusionConfig::v2_1(None, height, width);

        let vae = get_vae(None, &stable_diffusion_files::StableDiffusionVersion::V2_1, &sd_config, &Device::Cpu, DType::F16);
        assert!(vae.is_ok());

    }
}