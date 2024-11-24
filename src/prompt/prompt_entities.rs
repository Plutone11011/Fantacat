use std::fmt::Display;

#[derive(Debug, PartialEq, Clone)]
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
            Medium::OilPainting => write!(f, "oil painting"),
            Medium::Photography => write!(f, "photography"),
            Medium::PixelArt => write!(f, "pixel art"),
            Medium::DigitalArt => write!(f, "digital art"),
            Medium::LineArt => write!(f, "line art"),
            Medium::Print => write!(f, "print"),
            Medium::Comic => write!(f, "comic")
        }
        
    }
}


#[derive(Debug, PartialEq, Clone)]
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

#[derive(Debug, PartialEq, Clone)]
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
            Style::Hyperrealist => write!(f, "hyper realist"),
            Style::HighRes => write!(f, "high res"),
            Style::LowRes => write!(f, "low res")
        }
    }
}