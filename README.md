# ultratree-neural-baseline
This is a baseline of how a more traditional neural network would perform on the same data as ultratree

# Programs to ignore

- `test_evaluate_model.py` and `test_ffnn_senses.py` are AI-generated and
probably don't work.

# Programs to use

You will need a sense-annotated, prepared sqlite database.

So you have:
 
- A sense-annotated set of training stories

- You have run `ultrametric-trees/bin/prepare` on that, and have training data

- A sense-annotated set of held-out strories

- You have run `ultrametric-trees/bin/prepare` on that, and therefore have held-out data. 


You run `ffnn-senses.py --db-path` **training_data_file** 

It defaults to `--model-save-path model.pt` That will take a while to run. 

When it finishes, run `evaluate_model.py` with the following arguments:

-- `--model model.pt`  (or whatever file you put the model in)

-- `--input-db` (the held-out data)

-- `--output-db` (where to write it do)

-- `--description` (a description of what this run was, e.g. this might include the embedding dimension and hidden layer size if you specified these)

