use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::Result;
use itertools::Itertools;
use serde_json::Value;
use tracing::{debug, error, instrument};

#[allow(unused)]
#[derive(Debug, Copy, Clone)]
/// The dataset to load and work with.
pub enum Dataset {
    /// The movies dataset ~16MB
    Movies,
    /// A sample of the GHArchive dataset ~1.4GB
    GHArchive,
    /// Wikipedia abstract dataset ~6GB lots of indexable text.
    WikipediaAbstract,
}

impl FromStr for Dataset {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let s = s.to_lowercase();
        match s.as_str() {
            "movies" => Ok(Self::Movies),
            "gharchive" => Ok(Self::GHArchive),
            "wikiabstract" => Ok(Self::WikipediaAbstract),
            other => Err(format!("Unknown dataset: {other:?}")),
        }
    }
}

#[allow(unused)]
impl Dataset {
    pub fn file_glob(&self) -> &str {
        match self {
            Dataset::Movies => "./datasets/movies.json",
            Dataset::GHArchive => "./datasets/gharchive/*.json",
            Dataset::WikipediaAbstract => {
                "./datasets/enwiki-2024-09-20-abstract/*.ndjson"
            },
        }
    }

    pub fn reader_concurrency(&self) -> usize {
        match self {
            Dataset::Movies => 1,
            Dataset::GHArchive => 8,
            Dataset::WikipediaAbstract => 8,
        }
    }
}

#[allow(unused)]
/// Creates a new stream of document to ingest.
pub fn stream_dataset<PP, T>(
    dataset: Dataset,
    pre_processor: PP,
) -> Result<(flume::Receiver<T>, Arc<AtomicUsize>)>
where
    PP: Fn(serde_json::Map<String, Value>) -> T + Send + Clone + 'static,
    T: Send + 'static,
{
    let path = dataset.file_glob();
    let glob = glob::glob(path)?;

    let (tx, rx) = flume::bounded(1_000_000);
    let total_bytes_read = Arc::new(AtomicUsize::default());
    for chunk in &glob.chunks(dataset.reader_concurrency()) {
        let chunk = chunk.collect::<Result<Vec<PathBuf>, _>>()?;
        let tx = tx.clone();
        let total_bytes_read = total_bytes_read.clone();
        let pre_processor = pre_processor.clone();
        std::thread::spawn(move || {
            for file in chunk {
                match read_file(file, dataset, tx.clone(), pre_processor.clone()) {
                    Ok(read) => {
                        total_bytes_read.fetch_add(read, Ordering::Relaxed);
                    },
                    Err(e) => {
                        error!(error = ?e, "Failed to read file");
                    },
                }
            }
        });
    }

    Ok((rx, total_bytes_read))
}

#[instrument("ingest", skip(tx, pre_processor))]
fn read_file<PP, T>(
    path: PathBuf,
    dataset: Dataset,
    tx: flume::Sender<T>,
    pre_processor: PP,
) -> Result<usize>
where
    PP: Fn(serde_json::Map<String, Value>) -> T + Send + Clone + 'static,
    T: Send + 'static,
{
    let reader = BufReader::with_capacity(100 << 20, File::open(path)?);

    let mut bytes_read = 0;
    match dataset {
        Dataset::Movies => {
            let movies: Vec<serde_json::Map<String, Value>> =
                serde_json::from_reader(reader)?;
            for record in movies {
                let _ = tx.send(pre_processor(record));
            }
        },
        Dataset::GHArchive => {
            stream_ndjson(reader, &mut bytes_read, tx, pre_processor)?;
        },
        Dataset::WikipediaAbstract => {
            stream_ndjson(reader, &mut bytes_read, tx, pre_processor)?;
        },
    }

    Ok(bytes_read)
}

fn stream_ndjson<PP, T>(
    reader: BufReader<File>,
    bytes_read: &mut usize,
    tx: flume::Sender<T>,
    pre_processor: PP,
) -> Result<()>
where
    PP: Fn(serde_json::Map<String, Value>) -> T + Send + Clone + 'static,
    T: Send + 'static,
{
    let mut count = 0;
    let mut skipped = 0;
    for line in reader.lines() {
        let line = line?;
        let record = match serde_json::from_str::<serde_json::Map<String, Value>>(&line)
        {
            Err(_) => {
                skipped += 1;
                continue;
            },
            Ok(record) => record,
        };

        count += 1;
        *bytes_read += line.as_bytes().len();

        let _ = tx.send(pre_processor(record));
    }

    debug!(records = count, skipped = skipped, "Processed ingest file");

    Ok(())
}

/// Flattens the top level fields until it gets to a single value or array.
///
/// Objects like:
///
/// ```
/// {
///     "foo": {
///         "bar": [1, 2, 3],
///         "baz": "example"
///     }
/// }
/// ```
///
/// becomes:
///
/// ```
/// {
///     "foo.bar": [1, 2, 3],
///     "foo.baz": "example"
/// }
/// ```
pub fn flatten_top_level_fields(
    object: serde_json::Map<String, Value>,
) -> serde_json::Map<String, Value> {
    let mut keys = Vec::new();
    let mut new_object = serde_json::Map::with_capacity(object.len());

    for (key, value) in object {
        keys.push(key);
        flatten_object(&mut keys, &mut new_object, value);
        keys.pop();
    }

    new_object
}

fn flatten_object(
    keys: &mut Vec<String>,
    new_object: &mut serde_json::Map<String, Value>,
    value: Value,
) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                keys.push(key);
                flatten_object(keys, new_object, value);
                keys.pop();
            }
        },
        other => {
            let key = format_keys(keys.as_slice());
            new_object.insert(key, other);
        },
    }
}

fn format_keys(keys: &[String]) -> String {
    keys.join(".")
}
