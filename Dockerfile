# CPU image built directly from this checkout (Linux x86_64).
FROM rust:1.99.0-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /src
# Official ORT archive, verified against the release asset SHA-256.
RUN curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/onnxruntime-linux-x64-1.29.0.tgz -o /tmp/ort.tgz && \
    echo 'c3fddc4f139a045b0c4902c57410f0694f1c2fdf9b6939fbe38b1aeae7cd14ba  /tmp/ort.tgz' | sha256sum -c - && \
    mkdir -p /opt/ort && tar -xzf /tmp/ort.tgz --strip-components=1 -C /opt/ort && rm /tmp/ort.tgz
ENV BLUE_ONYX_ORT_LIB_DIR=/opt/ort/lib
COPY . .
RUN cargo test --locked --lib && cargo build --locked --release --bin blue_onyx

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl libssl3 && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --uid 1000 --create-home blueonyx && \
    mkdir -p /app/config /app/models && chown -R blueonyx:blueonyx /app
WORKDIR /app
COPY --from=builder /src/target/release/blue_onyx ./blue_onyx
COPY --from=builder /opt/ort/lib/ ./ort-libs/
COPY --from=builder /src/LICENSE ./LICENSE
COPY --from=builder /opt/ort/LICENSE /opt/ort/ThirdPartyNotices.txt ./notices/onnxruntime/
ENV ORT_DYLIB_PATH=/app/ort-libs/libonnxruntime.so \
    LD_LIBRARY_PATH=/app/ort-libs \
    BLUE_ONYX_CONFIG_PATH=/app/config/blue_onyx_config.json \
    BLUE_ONYX_MODEL_DIR=/app/models
USER blueonyx
EXPOSE 32168
HEALTHCHECK --interval=15s --timeout=3s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:32168/health/ready || exit 1
ENTRYPOINT ["./blue_onyx"]
