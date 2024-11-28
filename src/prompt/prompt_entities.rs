use std::fmt::Display;
use clap::ValueEnum;

#[derive(Debug, PartialEq, Clone, ValueEnum)]
pub enum Medium{
    OilPainting,
    Photography,
    PixelArt,
    Comic,
    DigitalArt,
    LineArt,
    Print
}

impl Display for Medium {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Medium::OilPainting => write!(f, "oil-painting"),
            Medium::Photography => write!(f, "photography"),
            Medium::PixelArt => write!(f, "pixel-art"),
            Medium::DigitalArt => write!(f, "digital-art"),
            Medium::LineArt => write!(f, "line-art"),
            Medium::Print => write!(f, "print"),
            Medium::Comic => write!(f, "comic")
        }
        
    }
}


#[derive(Debug, PartialEq, Clone, ValueEnum)]
pub enum Color{
    Gold,
    Silver,
    Red,
    White,
    Brown,
    Black,
    Grey
}

impl Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Color::Gold => write!(f, "gold"),
            Color::Silver => write!(f, "silver"),
            Color::Red => write!(f, "red"),
            Color::White => write!(f, "white"),
            Color::Brown => write!(f, "brown"),
            Color::Black => write!(f, "black"),
            Color::Grey => write!(f, "grey")
        }
    }
}

#[derive(Debug, PartialEq, Clone, ValueEnum)]
pub enum Style{
    Anime,
    Minimalist,
    Modern,
    Surrealist,
    Hyperrealist,
    HighRes,
    LowRes
}

impl Display for Style {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Style::Anime => write!(f, "anime"),
            Style::Minimalist => write!(f, "minimalist"),
            Style::Modern => write!(f, "modern"),
            Style::Surrealist => write!(f, "surrealist"),
            Style::Hyperrealist => write!(f, "hyper-realist"),
            Style::HighRes => write!(f, "high-res"),
            Style::LowRes => write!(f, "low-res")
        }
    }
}


#[derive(Debug, PartialEq, Clone, ValueEnum)]
pub enum Breed {
    MaineCoon,
    Abyssinian,
    Persian,
    Siamese,
    Bengal
}

impl Display for Breed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Breed::MaineCoon => write!(f, "maine-coon"),
            Breed::Abyssinian => write!(f, "abyssinian"),
            Breed::Bengal => write!(f, "bengal"),
            Breed::Persian => write!(f, "persian"),
            Breed::Siamese => write!(f, "siamese")
        }
    }
}