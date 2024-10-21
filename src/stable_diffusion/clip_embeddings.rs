use tokenizers::Tokenizer;
use candle_transformers::models::stable_diffusion;
use anyhow;
use candle_core;



use crate::stable_diffusion::stable_diffusion_files;

const CLIP_SPECIAL_TOKEN: &str = "<|endoftext|>";

fn get_tokenizer() -> anyhow::Result<Tokenizer>{

    let tokenizer_sd = stable_diffusion_files::StableDiffusionFiles::Tokenizer;

    let tokenizer_file = tokenizer_sd.get(None, true)?;

    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;

    Ok(tokenizer)
    
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

    let mut tokens = tokenizer
        .encode(text, true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    let padding_id = get_padding_id(&tokenizer, &stable_diffusion_config);
    while tokens.len() < stable_diffusion_config.clip.max_position_embeddings {
        tokens.push(padding_id)
    }
    let tokens = candle_core::Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
    Ok(tokens)

}


// pub fn get_embeddings(text :&str, tokenizer: &Tokenizer, stable_diffusion_config: &stable_diffusion::StableDiffusionConfig) -> anyhow::Result<Tensor>{

//     let pad_id = get_padding_id(tokenizer, stable_diffusion_config);

//     println!("Embedding prompt {}", text);

//     let mut tokens = tokenizer
//         .encode(text, true)
//         .map_err(anyhow::Error::msg)?
//         .get_ids()
//         .to_vec();

//     Ok(candle_core::Tensor(tokens))

// }


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer(){
        let tokenizer = get_tokenizer();

        assert!(tokenizer.is_ok())
    }
    // static TOKENIZER: anyhow::Result<Tokenizer> = get_tokenizer();
    #[test]
    fn test_get_padding_id() -> anyhow::Result<(), String>{

        let width = Some(640 as usize);
        let height: Option<usize> = Some(480 as usize);
        let sd_config = stable_diffusion::StableDiffusionConfig::v1_5(None, height, width);
        let tokenizer = get_tokenizer();

        match tokenizer {
            Ok(tokenizer) => {
                // assumes padding token in clip config has been set to None
                let padding_id = get_padding_id(&tokenizer, &sd_config);
                assert_eq!(padding_id, *tokenizer.get_vocab(true).get(CLIP_SPECIAL_TOKEN).unwrap());
                Ok(())
            },
            Err(_e) => Err(String::from("Error while downloading tokenizer"))
        }
    }

}
//     #[test]
//     fn get_padding_id() {