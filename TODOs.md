## Todos (By Priority)

- [ ] Add tests to make sure 100% the loss is correctly working.
- [ ] Modify or create a variant of the static span generator to accept a lowerK and upperK rather than a single K, it'll then for each position it'll generate spans of size from lowerK to upperK. lower and upper K can be derived from the dataset it self by looking at the shortest and longest prompt.
- [ ] Add some post processing on the decoding step to clean up the predictions and merge any two spans into a longer one that should be merged.
- [ ] Implement a segmentation and retrieval evaluation.
