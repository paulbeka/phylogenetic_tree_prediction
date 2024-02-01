FROM python:3.9

WORKDIR /app

# Copy your application code into the container
COPY . /app

# Install required packages
RUN apt-get update && apt-get install -y flex bison libgmp3-dev
RUN git clone --recursive https://github.com/amkozlov/raxml-ng /app/raxml-ng
RUN apt-get install cmake -y
RUN cd /app/raxml-ng && mkdir build && cd build && cmake -DUSE_MPI=ON .. && make

# Install Python requirements
RUN pip install -r /app/requirements.txt

# Set up the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Define a volume for the output directory
VOLUME /app/src/data/output

CMD ["/app/entrypoint.sh"]
