use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{Parser, Subcommand};

use std::path::PathBuf;

use kmer_embeddings::generator::*;
use kmer_embeddings::plotter::*;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// does testing things
    Train {
        /// lists test values
        #[arg(short, long)]
        dimensions: u8,
        #[arg(short, long)]
        kmer_size: u8,
    },
    PlotScores {
        #[arg(short, long)]
        kmer_size: u8,       
       
    }
}

fn main() {
    let args = Cli::parse();
    match args.command {
        Commands::Train {
            dimensions,
            kmer_size,
        } => {
            println!(
                "Training with dimensions: {} and kmer_size: {}",
                dimensions, kmer_size
            );
        },
        Commands::PlotScores {
            kmer_size,
        } => {
            plot(kmer_size);
        }
    }
}
