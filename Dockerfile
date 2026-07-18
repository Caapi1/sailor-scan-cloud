# Sailor scan-pipeline cloud image — Lane B (RunPod serverless, RTX 4090).
# ffmpeg + COLMAP 3.11.1 (CUDA SIFT, sm_89) + OpenSplat (CUDA, sm_89) + the
# RunPod handler. Versions are pinned to the combos the upstream projects
# CI-test: CUDA 12.1.1 + libtorch 2.2.1/cu121 (OpenSplat's documented pair —
# deviating is what breaks, see OpenSplat issue #131). COLMAP is pinned 3.x
# on purpose: 4.x bin layout misparses in OpenSplat (the Jul 17 Mac scar —
# solved here by never entering 4.x territory).
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake ninja-build build-essential wget unzip ca-certificates \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev \
    libglew-dev libcgal-dev libceres-dev libcurl4-openssl-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch 3.11.1 --depth 1 https://github.com/colmap/colmap.git /colmap \
    && cmake -S /colmap -B /colmap/build -GNinja -DCMAKE_BUILD_TYPE=Release \
       -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGUI_ENABLED=OFF \
    && ninja -C /colmap/build install

RUN wget -q "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.2.1%2Bcu121.zip" -O /lt.zip \
    && unzip -q /lt.zip -d /opt && rm /lt.zip

RUN git clone --depth 1 https://github.com/pierotofy/OpenSplat /os \
    && cmake -S /os -B /os/build -GNinja -DCMAKE_BUILD_TYPE=Release \
       -DGPU_RUNTIME=CUDA -DCMAKE_PREFIX_PATH=/opt/libtorch \
       -DCMAKE_CUDA_ARCHITECTURES=89 \
    && ninja -C /os/build

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg python3 python3-pip \
    libboost-program-options1.74.0 libboost-graph1.74.0 libflann1.9 \
    libfreeimage3 libmetis5 libgoogle-glog0v5 libsqlite3-0 libglew2.2 \
    libceres2 libcurl4 libgomp1 libgmp10 libmpfr6 \
    libopencv-core4.5d libopencv-imgproc4.5d libopencv-imgcodecs4.5d \
    libopencv-calib3d4.5d libopencv-highgui4.5d libopencv-videoio4.5d \
    libopencv-features2d4.5d libopencv-flann4.5d \
    && pip install --no-cache-dir runpod \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local /usr/local
COPY --from=build /os/build/opensplat /usr/local/bin/opensplat
COPY --from=build /opt/libtorch/lib /opt/libtorch/lib
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:/usr/local/bin:/usr/local/lib
COPY handler.py /app/handler.py
WORKDIR /app
CMD ["python3", "-u", "/app/handler.py"]
