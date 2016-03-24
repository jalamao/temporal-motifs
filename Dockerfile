FROM ubuntu:latest

MAINTAINER Andrew Mellor "mellor91@hotmail.co.uk"

RUN apt-get update -y
RUN apt-get install -f -y git \
					      wget \
					      libgraphviz-dev


RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda2-3.19.0-Linux-x86_64.sh && \
    /bin/bash /Miniconda2-3.19.0-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda2-3.19.0-Linux-x86_64.sh

#RUN wget http://www.graphviz.org/pub/graphviz/stable/ubuntu/ub12.04/x86_64/graphviz_2.38.0-1~precise_amd64.deb && \
#		apt-get install ./graphviz_2.38.0-1~precise_amd64.deb -f

# Eventually just pull the git repository
#RUN git clone https://github.com/empiricalstateofmind/personal_website.git ./app

ENV PATH /opt/conda/bin:$PATH

COPY . /motifs
WORKDIR /motifs

#RUN pip install pygraphviz
#RUN pip install -r ./environment.yml

#ENTRYPOINT ["python"]
#CMD ["app.py"]