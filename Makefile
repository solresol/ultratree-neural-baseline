SENSE_ANNOTATED_PREPPED_TRAINING_DATA=/ultratree/language-model/tiny.sqlite
SENSE_ANNOTATED_PREPPED_HELD_OUT_DATA=/ultratree/language-model/validation.sqlite
COMPRESSED_SENSE_ANNOTATED_TRAINING_DATA=ultratree-sense-annotated-training-data.sql.gz
COMPRESSED_SENSE_ANNOTATED_TRAINING_DECODINGS=ultratree-sense-annotated-training-decode.sql.gz
COMPRESSED_SENSE_ANNOTATED_HELD_OUT_DATA=ultratree-sense-annotated-heldout-data.sql.gz
COMPRESSED_SENSE_ANNOTATED_HELD_OUT_DECODINGS=ultratree-sense-annotated-training-heldout.sql.gz

.PHONY: all

all: $(COMPRESSED_SENSE_ANNOTATED_TRAINING_DATA) $(COMPRESSED_SENSE_ANNOTATED_TRAINING_DECODINGS) $(COMPRESSED_SENSE_ANNOTATED_HELD_OUT_DATA) $(COMPRESSED_SENSE_ANNOTATED_HELD_OUT_DECODINGS)
	echo Done

# Explicit rules for SQL file generation
training_data_training.sql: $(SENSE_ANNOTATED_PREPPED_TRAINING_DATA)
	@if [ ! -f $@ ]; then \
		echo "Creating $@..."; \
		sqlite3 $(SENSE_ANNOTATED_PREPPED_TRAINING_DATA) ".dump training_data" > $@; \
	fi

decodings_training.sql: $(SENSE_ANNOTATED_PREPPED_TRAINING_DATA)
	@if [ ! -f $@ ]; then \
		echo "Creating $@..."; \
		sqlite3 $(SENSE_ANNOTATED_PREPPED_TRAINING_DATA) ".dump decodings" > $@; \
	fi

training_data_heldout.sql: $(SENSE_ANNOTATED_PREPPED_HELD_OUT_DATA)
	@if [ ! -f $@ ]; then \
		echo "Creating $@..."; \
		sqlite3 $(SENSE_ANNOTATED_PREPPED_HELD_OUT_DATA) ".dump training_data" > $@; \
	fi

decodings_heldout.sql: $(SENSE_ANNOTATED_PREPPED_HELD_OUT_DATA)
	@if [ ! -f $@ ]; then \
		echo "Creating $@..."; \
		sqlite3 $(SENSE_ANNOTATED_PREPPED_HELD_OUT_DATA) ".dump decodings" > $@; \
	fi

# Rules for creating compressed files
$(COMPRESSED_SENSE_ANNOTATED_TRAINING_DATA): training_data_training.sql
	gzip -9 < $< > $@

$(COMPRESSED_SENSE_ANNOTATED_TRAINING_DECODINGS): decodings_training.sql
	gzip -9 < $< > $@

$(COMPRESSED_SENSE_ANNOTATED_HELD_OUT_DATA): training_data_heldout.sql
	gzip -9 < $< > $@

$(COMPRESSED_SENSE_ANNOTATED_HELD_OUT_DECODINGS): decodings_heldout.sql
	gzip -9 < $< > $@
