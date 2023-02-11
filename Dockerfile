FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ARG WORKDIR_PATH=/project
WORKDIR $WORKDIR_PATH

ADD requirements-no-torch.txt $WORKDIR_PATH
RUN pip3 install -r $WORKDIR_PATH/requirements-no-torch.txt

ADD . $WORKDIR_PATH

# Setting the run arguments can be done during docker run
CMD ['train.py']