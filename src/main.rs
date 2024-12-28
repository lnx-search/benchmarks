#[cfg(feature = "lnx-tantivy")]
mod bench_lnx_tantivy;
mod datasets;

use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use tracing::info;

use crate::datasets::Dataset;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Debug, Parser)]
struct Args {
    #[arg(short, long, default_value = "movies")]
    dataset: Dataset,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info,tantivy=warn,lnx_fs=warn,lnx_tantivy=warn");
    }

    tracing_subscriber::fmt::init();

    info!("running lnx benchmarks");

    #[cfg(feature = "lnx-tantivy")]
    bench_lnx_tantivy::main(args.dataset)?;

    info!("Allowing cleanup of resources before exist...");
    std::thread::sleep(Duration::from_secs(1));

    Ok(())
}
