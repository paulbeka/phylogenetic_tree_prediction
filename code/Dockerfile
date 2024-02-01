FROM python:3.9

WORKDIR /app

# Copy your application code into the container
COPY . /app

# Install required packages using apt-get
RUN apt-get update && \
    apt-get install -y flex bison libgmp3-dev cmake && \
    git clone --recursive https://github.com/amkozlov/raxml-ng /app/raxml-ng && \
    cd /app/raxml-ng && mkdir build && cd build && cmake -DUSE_MPI=ON .. && make && \
    mv bin/raxml-ng /bin/

# Install Python requirements
RUN pip install -r /app/requirements.txt

# Set up the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Define a volume for the output directory
VOLUME /app/src/data/output

CMD ["/app/entrypoint.sh"]
