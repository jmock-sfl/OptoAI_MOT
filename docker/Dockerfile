FROM nvcr.io/nvidia/pytorch:22.05-py3

RUN apt-get update 
#RUN apt-get clean
#RUN apt-get install libgl1
#RUN apt install libeigen3-dev
#RUN apt-get install libspdlog-dev
#RUN apt-get install 

RUN pip install -U pip
RUN pip install pycuda
RUN pip install onnxruntime
RUN pip install opencv-python==4.5.5.64 


# Install SOLO
RUN conda install cython -y && conda clean --all
RUN git clone https://github.com:WXinlong/SOLO.git /SOLO
WORKDIR /SOLO
RUN pip install --no-cache-dir -e .

# Install spdlog
RUN git clone https://github.com/gabime/spdlog.git
RUN cd spdlog && mkdir build && cd build
RUN cmake .. && make -j
RUN cd ../../
