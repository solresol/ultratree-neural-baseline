SENSE_ANNOTATED_TEST_DATAFRAME=../ultrametric-trees/sense-annotated-test-dataframe.sqlite
SENSE_ANNOTATED_TRAINING_DATAFRAME=../ultrametric-trees/sense-annotated-training-dataframe.sqlite

UNANNOTATED_TEST_DATAFRAME=../ultrametric-trees/unannotated-test-dataframe.sqlite
UNANNOTATED_TRAINING_DATAFRAME=../ultrametric-trees/unannotated-training-dataframe.sqlite

.PHONY: all train-models evaluate-models

all: train-models evaluate-models
	echo "All tasks complete."

# Train models with different embedding and hidden dimensions
train-models: sense_annotated_model_emb2_hidden4.pt sense_annotated_model_emb4_hidden8.pt sense_annotated_model_emb8_hidden16.pt sense_annotated_model_emb16_hidden32.pt sense_annotated_model_emb32_hidden64.pt sense_annotated_model_emb64_hidden128.pt sense_annotated_model_emb128_hidden256.pt unannotated_model_emb2_hidden4.pt unannotated_model_emb4_hidden8.pt unannotated_model_emb8_hidden16.pt unannotated_model_emb16_hidden32.pt unannotated_model_emb32_hidden64.pt unannotated_model_emb64_hidden128.pt unannotated_model_emb128_hidden256.pt
	echo "Model training complete."

sense_annotated_model_emb2_hidden4.pt: $(SENSE_ANNOTATED_TRAINING_DATAFRAME)
	python3 ffnn-senses.py --db-path $< --model-save-path $@ --embedding-dim 2 --hidden-dim 4

sense_annotated_model_emb4_hidden8.pt: $(SENSE_ANNOTATED_TRAINING_DATAFRAME)
	python3 ffnn-senses.py --db-path $< --model-save-path $@ --embedding-dim 4 --hidden-dim 8

sense_annotated_model_emb8_hidden16.pt: $(SENSE_ANNOTATED_TRAINING_DATAFRAME)
	python3 ffnn-senses.py --db-path $< --model-save-path $@ --embedding-dim 8 --hidden-dim 16

sense_annotated_model_emb16_hidden32.pt: $(SENSE_ANNOTATED_TRAINING_DATAFRAME)
	python3 ffnn-senses.py --db-path $< --model-save-path $@ --embedding-dim 16 --hidden-dim 32

sense_annotated_model_emb32_hidden64.pt: $(SENSE_ANNOTATED_TRAINING_DATAFRAME)
	python3 ffnn-senses.py --db-path $< --model-save-path $@ --embedding-dim 32 --hidden-dim 64

sense_annotated_model_emb64_hidden128.pt: $(SENSE_ANNOTATED_TRAINING_DATAFRAME)
	python3 ffnn-senses.py --db-path $< --model-save-path $@ --embedding-dim 64 --hidden-dim 128

sense_annotated_model_emb128_hidden256.pt: $(SENSE_ANNOTATED_TRAINING_DATAFRAME)
	python3 ffnn-senses.py --db-path $< --model-save-path $@ --embedding-dim 128 --hidden-dim 256


unannotated_model_emb2_hidden4.pt: $(UNANNOTATED_TRAINING_DATAFRAME)
	python3 ffnn-senses.py --db-path $< --model-save-path $@ --embedding-dim 2 --hidden-dim 4

unannotated_model_emb4_hidden8.pt: $(UNANNOTATED_TRAINING_DATAFRAME)
	python3 ffnn-senses.py --db-path $< --model-save-path $@ --embedding-dim 4 --hidden-dim 8

unannotated_model_emb8_hidden16.pt: $(UNANNOTATED_TRAINING_DATAFRAME)
	python3 ffnn-senses.py --db-path $< --model-save-path $@ --embedding-dim 8 --hidden-dim 16

unannotated_model_emb16_hidden32.pt: $(UNANNOTATED_TRAINING_DATAFRAME)
	python3 ffnn-senses.py --db-path $< --model-save-path $@ --embedding-dim 16 --hidden-dim 32

unannotated_model_emb32_hidden64.pt: $(UNANNOTATED_TRAINING_DATAFRAME)
	python3 ffnn-senses.py --db-path $< --model-save-path $@ --embedding-dim 32 --hidden-dim 64

unannotated_model_emb64_hidden128.pt: $(UNANNOTATED_TRAINING_DATAFRAME)
	python3 ffnn-senses.py --db-path $< --model-save-path $@ --embedding-dim 64 --hidden-dim 128

unannotated_model_emb128_hidden256.pt: $(UNANNOTATED_TRAINING_DATAFRAME)
	python3 ffnn-senses.py --db-path $< --model-save-path $@ --embedding-dim 128 --hidden-dim 256




# Evaluate all trained sense_annotated_models
evaluate-models: eval_sense_annotated_emb2_hidden4.out eval_sense_annotated_emb4_hidden8.out eval_sense_annotated_emb8_hidden16.out eval_sense_annotated_emb16_hidden32.out eval_sense_annotated_emb32_hidden64.out eval_sense_annotated_emb64_hidden128.out eval_sense_annotated_emb128_hidden256.out eval_unannotated_emb2_hidden4.out eval_unannotated_emb4_hidden8.out eval_unannotated_emb8_hidden16.out eval_unannotated_emb16_hidden32.out eval_unannotated_emb32_hidden64.out eval_unannotated_emb64_hidden128.out eval_unannotated_emb128_hidden256.out
	echo "Model evaluations complete."

eval_sense_annotated_emb2_hidden4.out: sense_annotated_model_emb2_hidden4.pt $(SENSE_ANNOTATED_TEST_DATAFRAME)
	evaluate_model.py --model sense_annotated_model_emb2_hidden4.pt --input-db $(SENSE_ANNOTATED_TEST_DATAFRAME) --output-db eval_$*.out --description "Sense Annotated, Embed=2, Hidden=4" > $@

eval_sense_annotated_emb4_hidden8.out: sense_annotated_model_emb4_hidden8.pt $(SENSE_ANNOTATED_TEST_DATAFRAME)
	evaluate_model.py --model sense_annotated_model_emb4_hidden8.pt --input-db $(SENSE_ANNOTATED_TEST_DATAFRAME) --output-db eval_$*.out --description "Sense Annotated, Embed=4, Hidden=8" > $@

eval_sense_annotated_emb8_hidden16.out: sense_annotated_model_emb8_hidden16.pt $(SENSE_ANNOTATED_TEST_DATAFRAME)
	evaluate_model.py --model sense_annotated_model_emb8_hidden16.pt --input-db $(SENSE_ANNOTATED_TEST_DATAFRAME) --output-db eval_$*.out --description "Sense Annotated, Embed=8, Hidden=16" > $@

eval_sense_annotated_emb16_hidden32.out: sense_annotated_model_emb16_hidden32.pt $(SENSE_ANNOTATED_TEST_DATAFRAME)
	evaluate_model.py --model sense_annotated_model_emb16_hidden32.pt --input-db $(SENSE_ANNOTATED_TEST_DATAFRAME) --output-db eval_$*.out --description "Sense Annotated, Embed=16, Hidden=32" > $@

eval_sense_annotated_emb32_hidden64.out: sense_annotated_model_emb32_hidden64.pt $(SENSE_ANNOTATED_TEST_DATAFRAME)
	evaluate_model.py --model sense_annotated_model_emb32_hidden64.pt --input-db $(SENSE_ANNOTATED_TEST_DATAFRAME) --output-db eval_$*.out --description "Sense Annotated, Embed=32, Hidden=64" > $@

eval_sense_annotated_emb64_hidden128.out: sense_annotated_model_emb64_hidden128.pt $(SENSE_ANNOTATED_TEST_DATAFRAME)
	evaluate_model.py --model sense_annotated_model_emb64_hidden128.pt --input-db $(SENSE_ANNOTATED_TEST_DATAFRAME) --output-db eval_$*.out --description "Sense Annotated, Embed=64, Hidden=128" > $@

eval_sense_annotated_emb128_hidden256.out: sense_annotated_model_emb128_hidden256.pt $(SENSE_ANNOTATED_TEST_DATAFRAME)
	evaluate_model.py --model sense_annotated_model_emb128_hidden256.pt --input-db $(SENSE_ANNOTATED_TEST_DATAFRAME) --output-db eval_$*.out --description "Sense Annotated, Embed=128, Hidden=256" > $@



eval_unannotated_emb2_hidden4.out: unannotated_model_emb2_hidden4.pt $(UNANNOTATED_TEST_DATAFRAME)
	evaluate_model.py --model unannotated_model_emb2_hidden4.pt --input-db $(UNANNOTATED_TEST_DATAFRAME) --output-db eval_$*.out --description "Unannotated, Embed=2, Hidden=4" > $@

eval_unannotated_emb4_hidden8.out: unannotated_model_emb4_hidden8.pt $(UNANNOTATED_TEST_DATAFRAME)
	evaluate_model.py --model unannotated_model_emb4_hidden8.pt --input-db $(UNANNOTATED_TEST_DATAFRAME) --output-db eval_$*.out --description "Unannotated, Embed=4, Hidden=8" > $@

eval_unannotated_emb8_hidden16.out: unannotated_model_emb8_hidden16.pt $(UNANNOTATED_TEST_DATAFRAME)
	evaluate_model.py --model unannotated_model_emb8_hidden16.pt --input-db $(UNANNOTATED_TEST_DATAFRAME) --output-db eval_$*.out --description "Unannotated, Embed=8, Hidden=16" > $@

eval_unannotated_emb16_hidden32.out: unannotated_model_emb16_hidden32.pt $(UNANNOTATED_TEST_DATAFRAME)
	evaluate_model.py --model unannotated_model_emb16_hidden32.pt --input-db $(UNANNOTATED_TEST_DATAFRAME) --output-db eval_$*.out --description "Unannotated, Embed=16, Hidden=32" > $@

eval_unannotated_emb32_hidden64.out: unannotated_model_emb32_hidden64.pt $(UNANNOTATED_TEST_DATAFRAME)
	evaluate_model.py --model unannotated_model_emb32_hidden64.pt --input-db $(UNANNOTATED_TEST_DATAFRAME) --output-db eval_$*.out --description "Unannotated, Embed=32, Hidden=64" > $@

eval_unannotated_emb64_hidden128.out: unannotated_model_emb64_hidden128.pt $(UNANNOTATED_TEST_DATAFRAME)
	evaluate_model.py --model unannotated_model_emb64_hidden128.pt --input-db $(UNANNOTATED_TEST_DATAFRAME) --output-db eval_$*.out --description "Unannotated, Embed=64, Hidden=128" > $@

eval_unannotated_emb128_hidden256.out: unannotated_model_emb128_hidden256.pt $(UNANNOTATED_TEST_DATAFRAME)
	evaluate_model.py --model unannotated_model_emb128_hidden256.pt --input-db $(UNANNOTATED_TEST_DATAFRAME) --output-db eval_$*.out --description "Unannotated, Embed=128, Hidden=256" > $@
