FROM continuumio/miniconda3

MAINTAINER Andrew Mellor "mellor91@hotmail.co.uk"

RUN apt-get update -y 
RUN apt-get upgrade -y
RUN apt-get install -y \
				      graphviz \
				      graphviz-dev \
				   	  pkg-config

# Eventually just pull the git repository
#RUN git clone https://github.com/empiricalstateofmind/motifs.git ./motifs

COPY . /motifs
WORKDIR /motifs

RUN pip install pygraphviz
#RUN pip install -r ./environment.yml

ENTRYPOINT ["bash"]
CMD ["while true; do ping 8.8.8.8; done"]