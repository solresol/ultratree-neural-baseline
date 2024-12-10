SENSE_ANNOTATED_PREPPED_TRAINING_DATA=/ultratree/language-model/tiny.sqlite
SENSE_ANNOTATED_PREPPED_HELD_OUT_DATA=/ultratree/language-model/validation.sqlite
COMPRESSED_SENSE_ANNOTATED_TRAINING_DATA=ultratree-sense-annotated-training-data.sql.gz
COMPRESSED_SENSE_ANNOTATED_TRAINING_DECODINGS=ultratree-sense-annotated-training-decode.sql.gz
COMPRESSED_SENSE_ANNOTATED_HELD_OUT_DATA=ultratree-sense-annotated-heldout-data.sql.gz
COMPRESSED_SENSE_ANNOTATED_HELD_OUT_DECODINGS=ultratree-sense-annotated-training-heldout.sql.gz

TRAINING_SQL=tiny_training_data.sql
VALIDATION_SQL=validation_training_data.sql
TRAINING_DB=tiny.sqlite
VALIDATION_DB=validation.sqlite

EMBEDDING_DIMS=$(shell seq 2 2 128)
HIDDEN_DIMS=$(shell seq 4 4 256)
MODELS=$(foreach emb,$(EMBEDDING_DIMS),$(foreach hid,$(HIDDEN_DIMS),model_$(emb)_$(hid).pt))

.PHONY: all recreate-dbs train-models evaluate-models

all: $(COMPRESSED_SENSE_ANNOTATED_TRAINING_DATA) \
     $(COMPRESSED_SENSE_ANNOTATED_TRAINING_DECODINGS) \
     $(COMPRESSED_SENSE_ANNOTATED_HELD_OUT_DATA) \
     $(COMPRESSED_SENSE_ANNOTATED_HELD_OUT_DECODINGS) \
     recreate-dbs \
     train-models \
     evaluate-models
	echo "All tasks complete."

# Recreate SQLite databases from SQL dumps
$(TRAINING_SQL): $(COMPRESSED_SENSE_ANNOTATED_TRAINING_DATA)
	gunzip -c $< > $@

$(TRAINING_DB): $(TRAINING_SQL)
	sqlite3 $@ < $<

$(VALIDATION_DB): $(VALIDATION_SQL)
	sqlite3 $@ < $<

recreate-dbs: $(TRAINING_DB) $(VALIDATION_DB)
	echo "Databases recreated."

# Train models with different embedding and hidden dimensions
train-models: $(MODELS)
	echo "Model training complete."

model_%.pt: $(TRAINING_DB)
	@emb=$$(echo $* | cut -d_ -f1); \
	hid=$$(echo $* | cut -d_ -f2); \
	echo "Training model with embedding_dim=$$emb hidden_dim=$$hid"; \
	ffnn-senses.py --db-path $< --model-save-path $@ --embedding-dim $$emb --hidden-dim $$hid

# Evaluate all trained models
evaluate-models: $(foreach emb,$(EMBEDDING_DIMS),$(foreach hid,$(HIDDEN_DIMS),evaluation_$(emb)_$(hid).out))
	echo "Model evaluation complete."

evaluation_%.out: model_%.pt $(VALIDATION_DB)
	@emb=$$(echo $* | cut -d_ -f1); \
	hid=$$(echo $* | cut -d_ -f2); \
	desc="Model with embedding_dim=$$emb and hidden_dim=$$hid"; \
	echo "Evaluating $$desc"; \
	evaluate_model.py --model $< --input-db $(VALIDATION_DB) --output-db evaluation_$*.sqlite --description "$$desc" > $@
