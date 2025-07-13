#!/bin/bash

# NOTE: this is to forward the port 7860 from a remote server to the local machine and thus be able to access the Gradio interface running on that port.
# The -N option is used to use the connection only for port forwarding, no shell access.
ssh -N -L 7860:localhost:7860 nadir@10.59.129.243