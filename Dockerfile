FROM python:3.10-slim

WORKDIR /app    

COPY requirements.txt ./
RUN apt-get update && apt-get install -y \
    apt-utils\
    gcc \
    g++ \
    libhdf5-dev \
    python3-dev\
    pkg-config\
    git

RUN pip install darts

RUN pip install --no-cache-dir numpy

RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8888
CMD [ "python","app.py" ]
