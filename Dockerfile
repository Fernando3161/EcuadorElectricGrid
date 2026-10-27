FROM mambaorg/micromamba:2.3.0@sha256:d008b2611d604c4e0c8e3652c7f9dca7cd65a103d68a2db30112702078a45c7a
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates bash coreutils findutils && rm -rf /var/lib/apt/lists/*
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/ecuador-docker-run
ENTRYPOINT ["/usr/local/bin/ecuador-docker-run"]
