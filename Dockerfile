FROM rust:1.83 AS builder

WORKDIR /usr/build

COPY src src
COPY Cargo.toml Cargo.toml
COPY Cargo.lock Cargo.lock

RUN cargo build --release --all-features

FROM ubuntu:24.04

WORKDIR /usr/app

COPY --from=builder /usr/build/target/release/lnx-benchmarks /usr/app/lnx-benchmarks

ENTRYPOINT ["/usr/app/lnx-benchmarks"]