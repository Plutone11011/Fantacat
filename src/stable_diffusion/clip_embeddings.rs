use candle_nn::Module;
use tokenizers::Tokenizer;
use candle_transformers::models::stable_diffusion;
use anyhow;
use candle_core::{Device, DType};


use crate::stable_diffusion::stable_diffusion_files;

const CLIP_SPECIAL_TOKEN: &str = "<|endoftext|>";

pub fn get_tokenizer(tokenizer_file: Option<String>, sd_version: &stable_diffusion_files::StableDiffusionVersion) -> anyhow::Result<Tokenizer>{

    let tokenizer = stable_diffusion_files::StableDiffusionFiles::Tokenizer;
    let sd = stable_diffusion_files::create_sd_from_version(sd_version);
    let tokenizer_file = sd.get(&tokenizer, tokenizer_file, true)?;

    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;

    Ok(tokenizer)
    
}

pub fn get_embedding_model(embedding_file: Option<String>, stable_diffusion_config: &stable_diffusion::StableDiffusionConfig, sd_version: &stable_diffusion_files::StableDiffusionVersion, device: &Device) -> anyhow::Result<stable_diffusion::clip::ClipTextTransformer>{
    let clip = stable_diffusion_files::StableDiffusionFiles::Clip;
    let sd = stable_diffusion_files::create_sd_from_version(sd_version);
    let clip_weights_file = sd.get(&clip, embedding_file, true)?;

    let text_model = stable_diffusion::build_clip_transformer(&stable_diffusion_config.clip, clip_weights_file, device, DType::F16)?;

    
    Ok(text_model)
}


fn get_padding_id(tokenizer: &Tokenizer, stable_diffusion_config: &stable_diffusion::StableDiffusionConfig) -> u32{
    // padding id depends on passed configuration
    let pad_id = match &stable_diffusion_config.clip.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get(CLIP_SPECIAL_TOKEN).unwrap()
    };
    pad_id
}

pub fn encode_prompt(prompt: &str, tokenizer: &Tokenizer, stable_diffusion_config: &stable_diffusion::StableDiffusionConfig, device: &candle_core::Device) -> anyhow::Result<candle_core::Tensor>{

    let tokens = tokenizer
        .encode(prompt, true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    let padding_id = get_padding_id(&tokenizer, &stable_diffusion_config);
    
    let padded_tokens: Vec<_>;
    let n_tokens = tokens.len();
    if  n_tokens <= stable_diffusion_config.clip.max_position_embeddings{
        padded_tokens = tokens
        .into_iter()
        .chain(std::iter::repeat(padding_id)
            .take(stable_diffusion_config.clip.max_position_embeddings - n_tokens))
        .collect();
    } 
    else {
        // prompt too long, panic
        anyhow::bail!("Prompt is too long ({}), max tokens allowed {}", n_tokens, stable_diffusion_config.clip.max_position_embeddings)
    }



    let encoded_prompt = candle_core::Tensor::new(padded_tokens.as_slice(), device)?.unsqueeze(0)?;
    Ok(encoded_prompt)

}


pub fn get_embeddings(encoded_prompt: &candle_core::Tensor, embedding_model: &stable_diffusion::clip::ClipTextTransformer) -> anyhow::Result<candle_core::Tensor>{

    let embeddings = embedding_model.forward(encoded_prompt)?;
    Ok(embeddings)

}

pub fn get_embeddings_for_guidance_scale(encoded_prompt: &candle_core::Tensor, encoded_uncond_prompt: &candle_core::Tensor, embedding_model: &stable_diffusion::clip::ClipTextTransformer) -> anyhow::Result<candle_core::Tensor>{
    let embeddings = embedding_model.forward(encoded_prompt)?;
    let uncond_embeddings = embedding_model.forward(&encoded_uncond_prompt)?;

    let final_embeddings = candle_core::Tensor::cat(&[uncond_embeddings, embeddings], 0)?;

    Ok(final_embeddings)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_diffusion_tokenizer(){
        let tokenizer = get_tokenizer(None, &stable_diffusion_files::StableDiffusionVersion::Turbo);

        assert!(tokenizer.is_ok())
    }
    #[test]
    fn stable_diffusion_get_padding_id() -> anyhow::Result<()>{

        let width = Some(640 as usize);
        let height: Option<usize> = Some(480 as usize);
        let sd_config = stable_diffusion::StableDiffusionConfig::sdxl(None, height, width);
        let tokenizer = get_tokenizer(None, &stable_diffusion_files::StableDiffusionVersion::Xl)?;
        // assumes padding token in clip config has been set to None
        let padding_id = get_padding_id(&tokenizer, &sd_config);
        assert_eq!(padding_id, *tokenizer.get_vocab(true).get("!").unwrap());
        Ok(())

    }

    #[test]
    fn stable_diffusion_encode_prompt() -> anyhow::Result<()>{
        let prompt = "Test sentence";

        let width = Some(640 as usize);
        let height: Option<usize> = Some(480 as usize);
        let sd_config = stable_diffusion::StableDiffusionConfig::v2_1(None, height, width);
        let tokenizer = get_tokenizer(None, &stable_diffusion_files::StableDiffusionVersion::V2_1)?;
            
        // assumes padding token in clip config has been set to None
        let encoded_prompt = encode_prompt(prompt, &tokenizer, &sd_config, &candle_core::Device::Cpu);
        assert!(encoded_prompt.is_ok());
        Ok(())
 
    }

    #[test]
    fn stable_diffusion_get_embedding_model() -> anyhow::Result<()>{

        let width = Some(640 as usize);
        let height: Option<usize> = Some(480 as usize);
        let sd_config = stable_diffusion::StableDiffusionConfig::v1_5(None, height, width);

        let embedding_model = get_embedding_model(None, &sd_config, &stable_diffusion_files::StableDiffusionVersion::V1_5, &candle_core::Device::Cpu);
        assert!(embedding_model.is_ok());
        Ok(())
 
    }

    #[test]
    fn stable_diffusion_get_embeddings() -> anyhow::Result<()>{

        let prompt: &str = "Test sentence";
        let uncond_prompt = "";

        let width = Some(640 as usize);
        let height: Option<usize> = Some(480 as usize);
        let sd_config = stable_diffusion::StableDiffusionConfig::sdxl_turbo(None, height, width);
        let tokenizer = get_tokenizer(None, &stable_diffusion_files::StableDiffusionVersion::Turbo)?;
            
        // assumes padding token in clip config has been set to None
        let encoded_prompt = encode_prompt(prompt, &tokenizer, &sd_config, &candle_core::Device::Cpu)?;
        let encoded_uncond_prompt = encode_prompt(uncond_prompt, &tokenizer, &sd_config, &candle_core::Device::Cpu)?;
        let embedding_model = get_embedding_model(None, &sd_config, &stable_diffusion_files::StableDiffusionVersion::Turbo, &candle_core::Device::Cpu)?;

        let embeddings = get_embeddings_for_guidance_scale(&encoded_prompt, &encoded_uncond_prompt, &embedding_model);

        assert!(embeddings.is_ok());
        
        if let Ok(embs) = embeddings {
            
            let embeddings_size: &candle_core::Shape = embs.shape();
            let encoded_prompt_size: &candle_core::Shape = encoded_prompt.shape();
            assert_eq!(embeddings_size.rank(), 3);
            assert_eq!(encoded_prompt_size.rank(), 2);
            assert_eq!(embeddings_size.clone().into_dims()[1], encoded_prompt_size.clone().into_dims()[1]);
            
        }
        // assert_eq!()
        Ok(())
    }


}