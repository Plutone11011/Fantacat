use candle_nn::Module;
use tokenizers::Tokenizer;
use candle_transformers::models::stable_diffusion;
use anyhow;
use candle_core::{Device, DType};


use crate::stable_diffusion::stable_diffusion_files;

const CLIP_SPECIAL_TOKEN: &str = "<|endoftext|>";

fn get_tokenizer(tokenizer_file: Option<String>) -> anyhow::Result<Tokenizer>{

    let tokenizer_sd = stable_diffusion_files::StableDiffusionFiles::Tokenizer;

    let tokenizer_file = tokenizer_sd.get(tokenizer_file, true)?;

    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;

    Ok(tokenizer)
    
}

fn get_embedding_model(embedding_file: Option<String>, stable_diffusion_config: &stable_diffusion::StableDiffusionConfig, device: &Device) -> anyhow::Result<stable_diffusion::clip::ClipTextTransformer>{
    let clip_sd = stable_diffusion_files::StableDiffusionFiles::Clip;

    let clip_weights_file = clip_sd.get(embedding_file, true)?;

    let text_model = stable_diffusion::build_clip_transformer(&stable_diffusion_config.clip, clip_weights_file, device, DType::F16)?;

    
    Ok(text_model)
}


fn get_padding_id(tokenizer: &Tokenizer, stable_diffusion_config: &stable_diffusion::StableDiffusionConfig) -> u32{
    // padding id depends on passed configuration
    let pad_id = match &stable_diffusion_config.clip.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get(CLIP_SPECIAL_TOKEN).unwrap()
    };
    println!("Padding token id is {}", pad_id);
    pad_id
}

fn encode_text(text: &str, tokenizer: &Tokenizer, stable_diffusion_config: &stable_diffusion::StableDiffusionConfig, device: &candle_core::Device) -> anyhow::Result<candle_core::Tensor>{

    let tokens = tokenizer
        .encode(text, true)
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
        // text too long, panic
        anyhow::bail!("Prompt is too long ({}), max tokens allowed {}", n_tokens, stable_diffusion_config.clip.max_position_embeddings)
    }



    let encoded_text = candle_core::Tensor::new(padded_tokens.as_slice(), device)?.unsqueeze(0)?;
    print!("{:?}", encoded_text);
    Ok(encoded_text)

}


pub fn get_embeddings(encoded_text :&candle_core::Tensor, embedding_model: &stable_diffusion::clip::ClipTextTransformer) -> anyhow::Result<candle_core::Tensor>{

    let embeddings = embedding_model.forward(encoded_text)?;
    println!("{:?}", embeddings);
    Ok(embeddings)

}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer(){
        let tokenizer = get_tokenizer(None);

        assert!(tokenizer.is_ok())
    }
    #[test]
    fn test_get_padding_id() -> anyhow::Result<()>{

        let width = Some(640 as usize);
        let height: Option<usize> = Some(480 as usize);
        let sd_config = stable_diffusion::StableDiffusionConfig::v1_5(None, height, width);
        let tokenizer = get_tokenizer(None)?;
        // assumes padding token in clip config has been set to None
        let padding_id = get_padding_id(&tokenizer, &sd_config);
        assert_eq!(padding_id, *tokenizer.get_vocab(true).get(CLIP_SPECIAL_TOKEN).unwrap());
        Ok(())

    }

    #[test]
    fn test_encode_text() -> anyhow::Result<()>{
        let text = "Test sentence";

        let width = Some(640 as usize);
        let height: Option<usize> = Some(480 as usize);
        let sd_config = stable_diffusion::StableDiffusionConfig::v1_5(None, height, width);
        let tokenizer = get_tokenizer(None)?;
            
        // assumes padding token in clip config has been set to None
        let encoded_text = encode_text(text, &tokenizer, &sd_config, &candle_core::Device::Cpu);
        assert!(encoded_text.is_ok());
        Ok(())
 
    }

    #[test]
    fn test_get_embedding_model() -> anyhow::Result<()>{

        let width = Some(640 as usize);
        let height: Option<usize> = Some(480 as usize);
        let sd_config = stable_diffusion::StableDiffusionConfig::v1_5(None, height, width);

        let embedding_model = get_embedding_model(None, &sd_config, &candle_core::Device::Cpu);
        assert!(embedding_model.is_ok());
        Ok(())
 
    }

    #[test]
    fn test_get_embeddings() -> anyhow::Result<()>{

        let text = "Test sentence";

        let width = Some(640 as usize);
        let height: Option<usize> = Some(480 as usize);
        let sd_config = stable_diffusion::StableDiffusionConfig::v1_5(None, height, width);
        let tokenizer = get_tokenizer(None)?;
            
        // assumes padding token in clip config has been set to None
        let encoded_text = encode_text(text, &tokenizer, &sd_config, &candle_core::Device::Cpu)?;
        let embedding_model = get_embedding_model(None, &sd_config, &candle_core::Device::Cpu)?;

        let embeddings = get_embeddings(&encoded_text, &embedding_model);

        assert!(embeddings.is_ok());
        
        if let Ok(embs) = embeddings {
            
            let embeddings_size: &candle_core::Shape = embs.shape();
            let encoded_text_size: &candle_core::Shape = encoded_text.shape();
            assert_eq!(embeddings_size.rank(), 3);
            assert_eq!(encoded_text_size.rank(), 2);
            assert_eq!(embeddings_size.clone().into_dims()[1], encoded_text_size.clone().into_dims()[1]);
            
        }
        // assert_eq!()
        Ok(())
    }


}