use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, Write};

use anyhow::{Context, Result};
use tracing::info;

static ENWIKI_INGEST_PATH: &str = "./datasets/enwiki-20240920-abstract.xml";
static ENWIKI_OUTPUT_PATH: &str = "./datasets/enwiki-2024-09-20-abstract";

pub fn main() -> Result<()> {
    info!("Preparing datasets");

    prepare_enwiki().context("Prepare ENWiki")?;

    Ok(())
}

#[tracing::instrument]
fn prepare_enwiki() -> Result<()> {
    info!("Parsing XML to NDJSON");

    let reader = BufReader::with_capacity(20 << 20, File::open(ENWIKI_INGEST_PATH)?);

    let items: WikipediaAbstractFeed = quick_xml::de::from_reader(reader)?;
    info!("Writing {} docs", items.doc.len());

    let mut part_n = 0;
    let mut num_docs = 0;
    let mut writer =
        create_writer(format!("{ENWIKI_OUTPUT_PATH}/part_{part_n}.ndjson"))?;
    for entry in items.doc {
        serde_json::to_writer(&mut writer, &entry)?;
        writer.write_all(b"\n")?;
        num_docs += 1;

        if num_docs >= 250_000 {
            num_docs = 0;
            part_n += 1;

            writer.flush()?;
            writer.get_mut().sync_all()?;
            writer =
                create_writer(format!("{ENWIKI_OUTPUT_PATH}/part_{part_n}.ndjson"))?;
        }
    }

    writer.flush()?;
    writer.get_mut().sync_all()?;

    Ok(())
}

fn create_writer(path: String) -> Result<BufWriter<File>> {
    let file = File::options()
        .create(true)
        .truncate(true)
        .write(true)
        .read(true)
        .open(path)?;
    let writer = BufWriter::new(file);
    Ok(writer)
}

#[derive(serde::Deserialize)]
struct WikipediaAbstractDataset {
    feed: WikipediaAbstractFeed,
}

#[derive(serde::Deserialize)]
struct WikipediaAbstractFeed {
    doc: Vec<WikipediaAbstract>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct WikipediaAbstract {
    title: String,
    url: String,
    #[serde(rename = "abstract")]
    text: String,
    links: WikipediaAbstractSubLinks,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct WikipediaAbstractSubLinks {
    sublink: Vec<WikipediaAbstractLink>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct WikipediaAbstractLink {
    anchor: String,
    link: String,
}
