# build off of tensorflow last CPU-only image
FROM tensorflow/tensorflow

# # create directory for Docker image to work from
# WORKDIR /dev

# # copy everything from working directory
# COPY . .

COPY reqs.txt reqs.txt

# install dependancies
RUN pip3 install -r reqs.txt
