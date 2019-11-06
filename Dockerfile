# enviornment for running analysis
FROM floydhub/dl-docker:cpu
ADD scripts ./scripts
ADD Makefile ./
RUN make analysis