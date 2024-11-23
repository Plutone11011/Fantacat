use candle_transformers::models::stable_diffusion::{self, unet_2d::UNet2DConditionModel};
use candle_core::{Device, DType};
use anyhow;

use crate::stable_diffusion::stable_diffusion_files;

pub fn get_unet(unet_file: Option<String>, stable_diffusion_config: &stable_diffusion::StableDiffusionConfig, device: &Device, dtype: DType, use_flash_attn: bool) -> anyhow::Result<UNet2DConditionModel>{

    let unet_sd = stable_diffusion_files::StableDiffusionFiles::Unet;

    let unet_weights_file = unet_sd.get(unet_file, true)?;

    let unet = stable_diffusion_config.build_unet(unet_weights_file, device, 4, use_flash_attn, dtype)?;

    Ok(unet)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unet() {

        let width = Some(640 as usize);
        let height: Option<usize> = Some(480 as usize);
        let sd_config = stable_diffusion::StableDiffusionConfig::v1_5(None, height, width);

        let unet = get_unet(None, &sd_config, &Device::Cpu, DType::F16, true);
        assert!(unet.is_ok());

    }
}