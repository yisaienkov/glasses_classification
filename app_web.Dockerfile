FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y curl ca-certificates sudo git bzip2 libx11-6 \
    libopencv-dev libgtk2.0-dev libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

ENV TZ=Europe/Kiev

COPY requirements_web.txt ./requirements.txt

RUN python -m pip install -U pip && \
    python -m pip install -r requirements.txt && \
    python -m pip cache purge

COPY /src/ /app/src/
COPY /resources/ /app/resources/

WORKDIR /app/

CMD streamlit run src/web.py --theme.base dark --server.port 5000
