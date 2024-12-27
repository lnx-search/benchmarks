use std::mem;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tracing::info;
use anyhow::{Context, Result};
use humansize::DECIMAL;
use lnx_fs::{RuntimeOptions, VirtualFileSystem};
use lnx_tantivy::{tantivy, LnxIndex};
use lnx_tantivy::tantivy::{Index, IndexSettings};
use lnx_tantivy::tantivy::indexer::IndexWriterOptions;
use lnx_tantivy::tantivy::merge_policy::NoMergePolicy;
use lnx_tantivy::tantivy::schema::{Schema, SchemaBuilder, FAST, INDEXED, STORED, STRING, TEXT};
use lnx_tantivy::tantivy::store::Compressor;
use crate::datasets::{Dataset};

pub fn main(dataset: Dataset) -> Result<()> {
    info!("Starting lnx-tantivy benchmarks");

    let schema = match dataset {
        Dataset::Movies => movies_schema(),
        Dataset::GHArchive => gharchive_schema(),
    };

    info!("Dataset schema has {} fields", schema.num_fields());

    index_tantivy(schema.clone(), dataset)
        .context("Index tantivy")?;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    
    rt.block_on(index_lnx_tantivy(schema, dataset))
        .context("Index lnx tantivy")?;
    
    Ok(())
}

fn index_tantivy(schema: Schema, dataset: Dataset) -> Result<()> {
    std::fs::create_dir_all("./data/lnx_fs_bench/tantivy_run/")?;

    let index = Index::create_in_dir("./data/lnx_fs_bench/tantivy_run/", schema)?;
    let schema = index.schema();
    
    let (rx, bytes_read) = crate::datasets::stream_dataset(dataset)
        .context("Read dataset")?;
    let (finished_tx, incoming) = flume::bounded(10000);
    for _ in 0..4 {
        let rx = rx.clone();
        let finished_tx = finished_tx.clone();
        let schema = schema.clone();
        std::thread::spawn(move || {
            let incoming = rx
                .into_iter()
                .map(crate::datasets::flatten_top_level_fields);

            for record in incoming {
                let doc = tantivy::TantivyDocument::from_json_object(&schema, record)?;                
                let _ = finished_tx.send(doc);
            }
            Ok::<_, anyhow::Error>(())
        });
    }
    drop(finished_tx);
    
    let options = IndexWriterOptions::builder()
        .num_worker_threads(4)
        .memory_budget_per_thread(100 << 20)
        .build();
    let mut writer: tantivy::IndexWriter = index.writer_with_options(options)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    
    let start = Instant::now();
    let mut num_docs = 0;
    for doc in incoming {
        writer.add_document(doc)?;        
        num_docs += 1;
    }    
    writer.commit()?;
    
    let elapsed = start.elapsed();
    let bytes_read = bytes_read.load(Ordering::Relaxed) as f32;
    let docs_sec = num_docs as f32 / elapsed.as_secs_f32();
    let bytes_sec = bytes_read / elapsed.as_secs_f32();
    info!(
        elapsed = ?elapsed, 
        num_docs = num_docs, 
        docs_per_sec = docs_sec, 
        bytes_read = humansize::format_size(bytes_read as u64, DECIMAL),
        throughput = humansize::format_size(bytes_sec as u64, DECIMAL),
        "Completed indexing",
    );
    
    Ok(())
}

async fn index_lnx_tantivy(schema: Schema, dataset: Dataset) -> Result<()> {
    std::fs::create_dir_all("./data/lnx_fs_bench/lnx_run/")?;

    let path = PathBuf::from("./data/lnx_fs_bench/lnx_run/");
    let rt_options = RuntimeOptions::builder().build();
    let vfs = VirtualFileSystem::mount(path, rt_options).await?;
    let bucket = vfs.create_bucket("dataset").await?;
    
    let (tx, bytes_read) = crate::datasets::stream_dataset(dataset)
        .context("Read dataset")?;
    
    let index = LnxIndex::create("dataset", bucket.clone(), schema).await?;
    let schema = index.schema();
    
    let (segments_tx, segments_rx) = flume::unbounded();
    const NUM_THREADS: usize = 4;
    for _ in 0.. NUM_THREADS {
        let tx = tx.clone();
        let index = index.clone();
        let segments_tx = segments_tx.clone();
        let schema = schema.clone();
        
        std::thread::spawn(move || {
            let incoming = tx
                .into_iter()
                .map(crate::datasets::flatten_top_level_fields);
            
            const BLOCK_SIZE: u64 = 250_000;
            
            let settings = IndexSettings {
                docstore_compression: Compressor::Lz4,
                docstore_compress_dedicated_thread: false,
                docstore_blocksize: 30 << 10,
            };
            
            let mut indexer = index.new_indexer_with_settings(settings.clone());
            let mut num_docs = 0;
            for record in incoming {
                let doc = tantivy::TantivyDocument::from_json_object(&schema, record)?;
                indexer.add_document(doc)?;
                num_docs += 1;
                
                if num_docs >= BLOCK_SIZE {
                    let new_indexer = index.new_indexer_with_settings(settings.clone());
                    let old_indexer = mem::replace(&mut indexer, new_indexer);
                    let segment = old_indexer.finish()?;
                    segments_tx.send(segment)?;
                    num_docs = 0;
                }
            }

            let segment = indexer.finish()?;
            segments_tx.send(segment)?;
            
            Ok::<_, anyhow::Error>(())
        });
    }
    drop(segments_tx);

    let start = Instant::now();
    let mut num_docs = 0;
    for segment in segments_rx {
        num_docs += segment.num_docs();
        index.add_segment(segment).await?;
    }

    let elapsed = start.elapsed();
    let bytes_read = bytes_read.load(Ordering::Relaxed) as f32;
    let docs_sec = num_docs as f32 / elapsed.as_secs_f32();
    let bytes_sec = bytes_read / elapsed.as_secs_f32();
    info!(
        elapsed = ?elapsed, 
        num_docs = num_docs, 
        docs_per_sec = docs_sec, 
        bytes_read = humansize::format_size(bytes_read as u64, DECIMAL),
        throughput = humansize::format_size(bytes_sec as u64, DECIMAL),
        "Completed indexing",
    );

    Ok(())
}

fn movies_schema() -> Schema {
    let mut schema_builder = SchemaBuilder::new();
    schema_builder.add_u64_field("id", FAST | STORED);
    schema_builder.add_text_field("title", TEXT | STORED);
    schema_builder.add_text_field("overview", TEXT | STORED);
    schema_builder.add_text_field("poster", STORED);
    schema_builder.add_text_field("genres", TEXT | FAST | STORED);
    schema_builder.add_i64_field("release_date", FAST | STORED);
    schema_builder.build()
}

fn gharchive_schema() -> Schema {
    let mut schema_builder = SchemaBuilder::new();
    schema_builder.add_text_field("id", STORED);

    add_schema_actor_struct("actor", &mut schema_builder);

    schema_builder.add_u64_field("repo.id", FAST | STORED);
    schema_builder.add_text_field("repo.name", TEXT | FAST | STORED);
    schema_builder.add_text_field("repo.url", STORED);

    schema_builder.add_bool_field("public", FAST | STORED);
    schema_builder.add_date_field("created_at", FAST | STORED);

    add_schema_actor_struct("org", &mut schema_builder);

    // Event payloads
    schema_builder.add_text_field("event_type", TEXT | FAST | STORED);

    // Push
    schema_builder.add_u64_field("event.push.push_id", STORED);
    schema_builder.add_u64_field("event.push.size", FAST | STORED);
    schema_builder.add_u64_field("event.push.distinct_size", FAST | STORED);
    schema_builder.add_text_field("event.push.ref", STRING | STORED);
    schema_builder.add_text_field("event.push.head", STRING | STORED);
    schema_builder.add_text_field("event.push.before", STRING | STORED);

    schema_builder.add_text_field("event.push.commits.sha", STRING | STORED);
    schema_builder.add_text_field("event.push.commits.message", TEXT | STORED);
    schema_builder.add_text_field("event.push.commits.author.email", STRING | FAST | STORED);
    schema_builder.add_text_field("event.push.commits.author.name", STRING | FAST | STORED);
    schema_builder.add_bool_field("event.push.commits.distinct", FAST | STORED);
    schema_builder.add_text_field("event.push.commits.url", STORED);

    // Create
    schema_builder.add_text_field("event.create.ref", STRING | STORED);
    schema_builder.add_text_field("event.create.ref_type", STRING | STORED);
    schema_builder.add_text_field("event.create.master_branch", STRING | STORED);
    schema_builder.add_text_field("event.create.description", TEXT | STORED);
    schema_builder.add_text_field("event.create.pusher_type", FAST | STRING | STORED);

    // Issue
    schema_builder.add_text_field("event.issues.action", FAST | STORED);
    add_schema_issue_struct("event.issues.issue", &mut schema_builder);

    // Issue Comment
    schema_builder.add_text_field("event.issue_comment.action", FAST | STORED);
    add_schema_issue_struct("event.issue_comment.issue", &mut schema_builder);
    add_schema_comment_struct("event.issue_comment.comment", &mut schema_builder);

    // PR
    schema_builder.add_text_field("event.pr.action", FAST | STORED);
    add_schema_pull_request_struct("event.pr.pull_request", &mut schema_builder);

    // PR Comment
    schema_builder.add_text_field("event.pr_comment.action", FAST | STORED);
    add_schema_pull_request_struct("event.pr_comment.pull_request", &mut schema_builder);
    add_schema_comment_struct("event.pr_comment.comment", &mut schema_builder);

    schema_builder.build()
}

fn add_schema_user_struct(prefix: &str, schema_builder: &mut SchemaBuilder) {
    schema_builder.add_u64_field( &format!("{prefix}.id"), FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.login"), TEXT | FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.gravatar_id"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.avatar_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.followers_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.following_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.gists_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.starred_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.subscriptions_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.organizations_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.repos_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.events_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.received_events_url"), STORED);
    schema_builder.add_bool_field(&format!("{prefix}.site_admin"), FAST | STORED);
}

fn add_schema_actor_struct(prefix: &str, schema_builder: &mut SchemaBuilder) {
    schema_builder.add_u64_field( &format!("{prefix}.id"), FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.login"), TEXT | FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.gravatar_id"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.avatar_url"), STORED);
}

fn add_schema_milestone_struct(prefix: &str, schema_builder: &mut SchemaBuilder) {
    schema_builder.add_u64_field( &format!("{prefix}.id"), FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.labels_url"), STORED);
    schema_builder.add_u64_field( &format!("{prefix}.number"), FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.title"), TEXT | FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.description"), TEXT | STORED);

    schema_builder.add_u64_field( &format!("{prefix}.open_issues"), FAST | STORED);
    schema_builder.add_u64_field( &format!("{prefix}.closed_issues"), FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.state"), STRING | FAST | STORED);
    schema_builder.add_date_field(&format!("{prefix}.created_at"), INDEXED | FAST | STORED);
    schema_builder.add_date_field(&format!("{prefix}.updated_at"), INDEXED | FAST | STORED);
    schema_builder.add_date_field(&format!("{prefix}.due_on"), INDEXED | FAST | STORED);
    schema_builder.add_date_field(&format!("{prefix}.closed_at"), INDEXED | FAST | STORED);

    add_schema_user_struct(&format!("{prefix}.creator"), schema_builder);
}

fn add_schema_issue_struct(prefix: &str, schema_builder: &mut SchemaBuilder) {
    schema_builder.add_u64_field( &format!("{prefix}.id"), FAST | STORED);
    schema_builder.add_u64_field( &format!("{prefix}.number"), FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.title"), TEXT | FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.labels_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.comments_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.events_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.html_url"), STORED);

    add_schema_user_struct(&format!("{prefix}.user"), schema_builder);

    schema_builder.add_text_field(&format!("{prefix}.labels.url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.labels.name"), STRING | FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.labels.color"), STORED);

    schema_builder.add_text_field(&format!("{prefix}.state"), STRING | FAST | STORED);
    schema_builder.add_bool_field(&format!("{prefix}.locked"), INDEXED | FAST | STORED);
    schema_builder.add_u64_field( &format!("{prefix}.comments"), FAST | STORED);
    schema_builder.add_date_field(&format!("{prefix}.created_at"), INDEXED | FAST | STORED);
    schema_builder.add_date_field(&format!("{prefix}.updated_at"), INDEXED | FAST | STORED);
    schema_builder.add_date_field(&format!("{prefix}.closed_at"), INDEXED | FAST | STORED);

    add_schema_user_struct(&format!("{prefix}.assignee"), schema_builder);
    add_schema_milestone_struct(&format!("{prefix}.milestone"), schema_builder);

    schema_builder.add_text_field(&format!("{prefix}.pull_request.url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.pull_request.html_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.pull_request.diff_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.pull_request.patch_url"), STORED);

    schema_builder.add_text_field(&format!("{prefix}.body"), TEXT | STORED);
}

fn add_schema_comment_struct(prefix: &str, schema_builder: &mut SchemaBuilder) {
    schema_builder.add_u64_field( &format!("{prefix}.id"), STORED);
    schema_builder.add_text_field( &format!("{prefix}.url"), STORED);
    schema_builder.add_text_field( &format!("{prefix}.html_url"), STORED);
    schema_builder.add_text_field( &format!("{prefix}.issue_url"), STORED);

    add_schema_user_struct(&format!("{prefix}.user"), schema_builder);

    schema_builder.add_date_field(&format!("{prefix}.created_at"), INDEXED | FAST | STORED);
    schema_builder.add_date_field(&format!("{prefix}.updated_at"), INDEXED | FAST | STORED);

    schema_builder.add_text_field(&format!("{prefix}.body"), TEXT | STORED);
}

fn add_schema_pull_request_struct(prefix: &str, schema_builder: &mut SchemaBuilder) {
    schema_builder.add_u64_field( &format!("{prefix}.id"), STORED);
    schema_builder.add_u64_field( &format!("{prefix}.number"), FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.title"), TEXT | FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.html_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.diff_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.patch_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.state"), FAST | STORED);
    schema_builder.add_bool_field(&format!("{prefix}.locked"), FAST | STORED);

    add_schema_user_struct(&format!("{prefix}.user"), schema_builder);

    schema_builder.add_text_field(&format!("{prefix}.body"), TEXT | STORED);

    schema_builder.add_date_field(&format!("{prefix}.created_at"), INDEXED | FAST | STORED);
    schema_builder.add_date_field(&format!("{prefix}.updated_at"), INDEXED | FAST | STORED);
    schema_builder.add_date_field(&format!("{prefix}.closed_at"), INDEXED | FAST | STORED);
    schema_builder.add_date_field(&format!("{prefix}.merged_at"), INDEXED | FAST | STORED);

    add_schema_user_struct(&format!("{prefix}.assignee"), schema_builder);
    add_schema_milestone_struct(&format!("{prefix}.milestone"), schema_builder);
}