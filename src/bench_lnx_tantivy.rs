use std::collections::VecDeque;
use std::mem;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use humansize::DECIMAL;
use lnx_fs::{RuntimeOptions, VirtualFileSystem};
use lnx_tantivy::indexer::SegmentMemory;
use lnx_tantivy::tantivy::indexer::IndexWriterOptions;
use lnx_tantivy::tantivy::merge_policy::NoMergePolicy;
use lnx_tantivy::tantivy::schema::{
    JsonObjectOptions,
    Schema,
    SchemaBuilder,
    TextFieldIndexing,
    FAST,
    INDEXED,
    STORED,
    STRING,
    TEXT,
};
use lnx_tantivy::tantivy::{Index, IndexSettings};
use lnx_tantivy::{tantivy, LnxIndex};
use tracing::{error, info};

use crate::datasets::Dataset;

const NUM_THREADS: usize = 4;
const MEMORY_LIMIT_PER_SEG: usize = 50 << 20;
const DOCS_PER_COMMIT: usize = 200_000;

pub fn main(dataset: Dataset) -> Result<()> {
    info!("Starting lnx-tantivy benchmarks");

    let schema = match dataset {
        Dataset::Movies => movies_schema(),
        Dataset::GHArchive => gharchive_schema(),
        Dataset::WikipediaAbstract => wikipedia_abstract_schema(),
    };

    info!("Dataset schema has {} fields", schema.num_fields());

    index_tantivy(schema.clone(), dataset).context("Index tantivy")?;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    rt.block_on(index_lnx_tantivy(schema, dataset))
        .context("Index lnx tantivy")?;

    std::thread::sleep(Duration::from_secs(1));

    let _ = std::fs::remove_dir_all("./data/lnx_fs_bench/tantivy_run/");
    let _ = std::fs::remove_dir_all("./data/lnx_fs_bench/lnx_run/");

    Ok(())
}

fn index_tantivy(schema: Schema, dataset: Dataset) -> Result<()> {
    std::fs::create_dir_all("./data/lnx_fs_bench/tantivy_run/")?;

    let index = Index::create_in_dir("./data/lnx_fs_bench/tantivy_run/", schema)?;
    let schema = index.schema();

    let schema_clone = schema.clone();
    let pre_processor = move |record| {
        let flattened = crate::datasets::flatten_top_level_fields(record);
        let doc = tantivy::TantivyDocument::from_json_object(&schema_clone, flattened).unwrap();
        dbg!(doc.len());
        doc
    };

    let (rx, bytes_read) = crate::datasets::stream_dataset(dataset, pre_processor)
        .context("Read dataset")?;

    let options = IndexWriterOptions::builder()
        .num_worker_threads(NUM_THREADS)
        .memory_budget_per_thread(MEMORY_LIMIT_PER_SEG)
        .build();
    let mut writer: tantivy::IndexWriter = index.writer_with_options(options)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));

    let mut num_docs = 0;
    let mut execute_time = Duration::default();
    let mut commit_time = Duration::default();
    let start = Instant::now();
    for doc in rx {
        let s1 = Instant::now();
        writer.add_document(doc)?;
        execute_time += s1.elapsed();
        num_docs += 1;

        if (num_docs % DOCS_PER_COMMIT) == 0 {
            let s1 = Instant::now();
            writer.commit()?;
            commit_time += s1.elapsed();
        }
    }
    let s1 = Instant::now();
    writer.commit()?;
    commit_time += s1.elapsed();

    let total_time = start.elapsed();
    display_results(
        total_time,
        execute_time,
        commit_time,
        bytes_read.load(Ordering::Relaxed),
        num_docs,
    );

    Ok(())
}

async fn index_lnx_tantivy(schema: Schema, dataset: Dataset) -> Result<()> {
    std::fs::create_dir_all("./data/lnx_fs_bench/lnx_run/")?;

    let path = PathBuf::from("./data/lnx_fs_bench/lnx_run/");
    let rt_options = RuntimeOptions::builder().build();
    let vfs = VirtualFileSystem::mount(path, rt_options).await?;
    let bucket = vfs.create_bucket("dataset").await?;

    let index = LnxIndex::create("dataset", bucket.clone(), schema).await?;
    let schema = index.schema();

    let schema_clone = schema.clone();
    let pre_processor = move |record| {
        let flattened = crate::datasets::flatten_top_level_fields(record);
        tantivy::TantivyDocument::from_json_object(&schema_clone, flattened).unwrap()
    };
    let (rx, bytes_read) = crate::datasets::stream_dataset(dataset, pre_processor)
        .context("Read dataset")?;

    let (segments_tx, segments_rx) = flume::bounded(4);
    for _ in 0..NUM_THREADS {
        spawn_lnx_indexing_worker(index.clone(), rx.clone(), segments_tx.clone());
    }
    drop(segments_tx);

    let mut num_docs = 0;
    let mut commit_time = Duration::default();
    let mut handles = VecDeque::new();
    let start = Instant::now();
    for seg in segments_rx {
        num_docs += seg.num_docs();
        let index = index.clone();
        let handle = tokio::spawn(async move {
            index.add_segment(seg).await
        });
        
        handles.push_back(handle);
        
        if handles.len() > 16 {
            let s1 = Instant::now();
            if let Some(handle) = handles.pop_front() {
                handle.await??;
            }
            commit_time += s1.elapsed();
        }
    }

    let s1 = Instant::now();
    for handle in handles {
        handle.await??;        
    }
    commit_time += s1.elapsed();
    
    let total_time = start.elapsed();
    display_results(
        total_time,
        Duration::default(),
        commit_time,
        bytes_read.load(Ordering::Relaxed),
        num_docs,
    );

    Ok(())
}

fn spawn_lnx_indexing_worker(
    index: LnxIndex,
    incoming: flume::Receiver<tantivy::TantivyDocument>,
    finished_segments: flume::Sender<SegmentMemory>,
) {
    let runner = move || {
        let mut indexer = index.new_indexer_with_settings(IndexSettings::default());

        let mut num_docs = 0;
        for doc in incoming {
            indexer.add_document(doc)?;
            num_docs += 1;

            if indexer.memory_usage() >= MEMORY_LIMIT_PER_SEG
                || (num_docs % (DOCS_PER_COMMIT / NUM_THREADS)) == 0
            {
                let new_indexer =
                    index.new_indexer_with_settings(IndexSettings::default());
                let old_indexer = mem::replace(&mut indexer, new_indexer);
                let segment = old_indexer.finish()?;
                finished_segments.send(segment)?;
            }
        }

        let segment = indexer.finish()?;
        finished_segments.send(segment)?;

        Ok::<_, anyhow::Error>(())
    };

    let wrapped = move || {
        if let Err(e) = runner() {
            error!(error = ?e, "Failed to run indexer");
        }
    };

    std::thread::spawn(wrapped);
}

fn display_results(
    total_time: Duration,
    true_time: Duration,
    commit_time: Duration,
    bytes_read: usize,
    total_docs: usize,
) {
    let stall_time = total_time - (true_time + commit_time);
    let bytes_sec = bytes_read as f32 / total_time.as_secs_f32();
    let docs_sec = total_docs as f32 / total_time.as_secs_f32();

    let formatted_bytes_sec =
        format!("{}/s", humansize::format_size(bytes_sec as u64, DECIMAL));
    let formatted_docs_sec = format!("{docs_sec}/s");

    info!(
        total_time = ?total_time,
        execution_time = ?true_time,
        commit_time = ?commit_time,
        stall_time = ?stall_time,
        bytes_rate = %formatted_bytes_sec,
        docs_rate = %formatted_docs_sec,
        "Finished indexing run"
    );
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

fn wikipedia_abstract_schema() -> Schema {
    let mut schema_builder = SchemaBuilder::new();
    schema_builder.add_text_field("title", TEXT | STORED);
    schema_builder.add_text_field("url", STORED);
    schema_builder.add_text_field("abstract", TEXT | STORED);
    schema_builder.add_json_field(
        "links.sublink",
        JsonObjectOptions::default()
            .set_stored()
            .set_fast(Some("raw"))
            .set_indexing_options(TextFieldIndexing::default().set_tokenizer("raw")),
    );
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
    schema_builder
        .add_text_field("event.push.commits.author.email", STRING | FAST | STORED);
    schema_builder
        .add_text_field("event.push.commits.author.name", STRING | FAST | STORED);
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
    schema_builder.add_u64_field(&format!("{prefix}.id"), FAST | STORED);
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
    schema_builder.add_u64_field(&format!("{prefix}.id"), FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.login"), TEXT | FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.gravatar_id"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.avatar_url"), STORED);
}

fn add_schema_milestone_struct(prefix: &str, schema_builder: &mut SchemaBuilder) {
    schema_builder.add_u64_field(&format!("{prefix}.id"), FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.labels_url"), STORED);
    schema_builder.add_u64_field(&format!("{prefix}.number"), FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.title"), TEXT | FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.description"), TEXT | STORED);

    schema_builder.add_u64_field(&format!("{prefix}.open_issues"), FAST | STORED);
    schema_builder.add_u64_field(&format!("{prefix}.closed_issues"), FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.state"), STRING | FAST | STORED);
    schema_builder
        .add_date_field(&format!("{prefix}.created_at"), INDEXED | FAST | STORED);
    schema_builder
        .add_date_field(&format!("{prefix}.updated_at"), INDEXED | FAST | STORED);
    schema_builder.add_date_field(&format!("{prefix}.due_on"), INDEXED | FAST | STORED);
    schema_builder
        .add_date_field(&format!("{prefix}.closed_at"), INDEXED | FAST | STORED);

    add_schema_user_struct(&format!("{prefix}.creator"), schema_builder);
}

fn add_schema_issue_struct(prefix: &str, schema_builder: &mut SchemaBuilder) {
    schema_builder.add_u64_field(&format!("{prefix}.id"), FAST | STORED);
    schema_builder.add_u64_field(&format!("{prefix}.number"), FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.title"), TEXT | FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.labels_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.comments_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.events_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.html_url"), STORED);

    add_schema_user_struct(&format!("{prefix}.user"), schema_builder);

    schema_builder.add_text_field(&format!("{prefix}.labels.url"), STORED);
    schema_builder
        .add_text_field(&format!("{prefix}.labels.name"), STRING | FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.labels.color"), STORED);

    schema_builder.add_text_field(&format!("{prefix}.state"), STRING | FAST | STORED);
    schema_builder.add_bool_field(&format!("{prefix}.locked"), INDEXED | FAST | STORED);
    schema_builder.add_u64_field(&format!("{prefix}.comments"), FAST | STORED);
    schema_builder
        .add_date_field(&format!("{prefix}.created_at"), INDEXED | FAST | STORED);
    schema_builder
        .add_date_field(&format!("{prefix}.updated_at"), INDEXED | FAST | STORED);
    schema_builder
        .add_date_field(&format!("{prefix}.closed_at"), INDEXED | FAST | STORED);

    add_schema_user_struct(&format!("{prefix}.assignee"), schema_builder);
    add_schema_milestone_struct(&format!("{prefix}.milestone"), schema_builder);

    schema_builder.add_text_field(&format!("{prefix}.pull_request.url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.pull_request.html_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.pull_request.diff_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.pull_request.patch_url"), STORED);

    schema_builder.add_text_field(&format!("{prefix}.body"), TEXT | STORED);
}

fn add_schema_comment_struct(prefix: &str, schema_builder: &mut SchemaBuilder) {
    schema_builder.add_u64_field(&format!("{prefix}.id"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.html_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.issue_url"), STORED);

    add_schema_user_struct(&format!("{prefix}.user"), schema_builder);

    schema_builder
        .add_date_field(&format!("{prefix}.created_at"), INDEXED | FAST | STORED);
    schema_builder
        .add_date_field(&format!("{prefix}.updated_at"), INDEXED | FAST | STORED);

    schema_builder.add_text_field(&format!("{prefix}.body"), TEXT | STORED);
}

fn add_schema_pull_request_struct(prefix: &str, schema_builder: &mut SchemaBuilder) {
    schema_builder.add_u64_field(&format!("{prefix}.id"), STORED);
    schema_builder.add_u64_field(&format!("{prefix}.number"), FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.title"), TEXT | FAST | STORED);
    schema_builder.add_text_field(&format!("{prefix}.url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.html_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.diff_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.patch_url"), STORED);
    schema_builder.add_text_field(&format!("{prefix}.state"), FAST | STORED);
    schema_builder.add_bool_field(&format!("{prefix}.locked"), FAST | STORED);

    add_schema_user_struct(&format!("{prefix}.user"), schema_builder);

    schema_builder.add_text_field(&format!("{prefix}.body"), TEXT | STORED);

    schema_builder
        .add_date_field(&format!("{prefix}.created_at"), INDEXED | FAST | STORED);
    schema_builder
        .add_date_field(&format!("{prefix}.updated_at"), INDEXED | FAST | STORED);
    schema_builder
        .add_date_field(&format!("{prefix}.closed_at"), INDEXED | FAST | STORED);
    schema_builder
        .add_date_field(&format!("{prefix}.merged_at"), INDEXED | FAST | STORED);

    add_schema_user_struct(&format!("{prefix}.assignee"), schema_builder);
    add_schema_milestone_struct(&format!("{prefix}.milestone"), schema_builder);
}
