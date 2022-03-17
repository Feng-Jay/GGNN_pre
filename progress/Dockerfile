FROM pytorch/pytorch:0.4_cuda9_cudnn7
# comment out the following two lines if you don't need these environment variables for network proxy
ENV http_proxy http://wwwcache.open.ac.uk:80
ENV https_proxy http://wwwcache.open.ac.uk:80
# add package for progress indicator
RUN pip install tqdm pyarrow
RUN pip install -U pip
RUN pip install tensorboardX sklearn
