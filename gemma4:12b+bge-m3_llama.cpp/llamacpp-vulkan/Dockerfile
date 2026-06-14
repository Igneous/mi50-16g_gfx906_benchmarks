# llama.cpp + Vulkan (mesa RADV) runtime for gfx906 — comparison rig vs the ROCm fork.
#
# WHY THIS EXISTS: the official ghcr.io/ggml-org/llama.cpp:server-vulkan image is
# BROKEN — its libggml-vulkan.so has an undefined symbol from an incomplete
# shader-gen build (issue #20868; the docs note GPU images aren't CI-tested), so
# ggml reports "no usable GPU found". The *release tarball* of the same build is
# intact, so we wrap those binaries in a minimal ubuntu + mesa-RADV runtime.
#
# Build (pin LLAMA_BUILD to match the ROCm fork's build for a clean A/B):
#   podman build --build-arg LLAMA_BUILD=b9623 -t localhost/llamacpp-vulkan-b9623 llamacpp-vulkan/
# Run via docker-compose.llamacpp-vulkan.yml (mounts the same models.ini + GGUFs).

FROM ubuntu:24.04 AS fetch
ARG LLAMA_BUILD=b9623
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -fsSL -o /tmp/v.tgz \
      "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_BUILD}/llama-${LLAMA_BUILD}-bin-ubuntu-vulkan-x64.tar.gz" && \
    mkdir -p /app && tar xzf /tmp/v.tgz -C /app --strip-components=1

FROM ubuntu:24.04
# libvulkan1 + mesa-vulkan-drivers = the RADV ICD (arch-agnostic, runs gfx906).
# libgomp1/libcurl4t64 are the binaries' runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 mesa-vulkan-drivers libgomp1 libcurl4t64 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY --from=fetch /app /app
ENV LD_LIBRARY_PATH=/app
WORKDIR /app
ENTRYPOINT ["/app/llama-server"]
