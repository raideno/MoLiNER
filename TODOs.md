## Todos (By Priority)

- [ ] Make visualizations only save forward output without .html visualizations
- [ ] Add tests to make sure 100% the loss is correctly working.
- [ ] Add some post processing on the decoding step to clean up the predictions and merge any two spans into a longer one that should be merged.
- [ ] In the case of a frozen prompts token encoder; we should enable some caching mechanism to cache the encoding of the prompts since they aren't going to change. And thus after caching we just drop the model from the memory.
