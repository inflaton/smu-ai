.PHONY: start
start:
	python app.py

serve:
ifeq ("$(PORT)", "")
	JINA_HIDE_SURVEY=1 TRANSFORMERS_OFFLINE=1 python -m lcserve deploy local server
else
	JINA_HIDE_SURVEY=1 TRANSFORMERS_OFFLINE=1 python -m lcserve deploy local server --port=${PORT}
endif

test:
	python test.py

test2:
	python server.py

chat:
	python test.py chat

chat2:
	python unit_test.py chat

unittest:
	python unit_test.py $(TEST)

tele:
	python telegram_bot.py

openllm:
ifeq ("$(PORT)", "")
	openllm start llama --model-id meta-llama/Llama-2-7b-chat-hf
else
	openllm start llama --model-id meta-llama/Llama-2-7b-chat-hf --port=${PORT}
endif

openllm-cpu:
	CUDA_VISIBLE_DEVICES="" openllm start llama --model-id meta-llama/Llama-2-7b-chat-hf

ingest:
	python ingest.py

mlock:
	@echo 'To set new value for mlock, please run: sudo prlimit --memlock=35413752832:35413752832 --pid $$$$'
	prlimit --memlock

.PHONY: format
format:
	isort .
	black .

install:
	pip install -U -r requirements.txt
	pip show langchain transformers
	
install-extra:
	CXX=g++-11  CC=gcc-11 pip install -U -r requirements_extra.txt
	pip show llama-cpp-python ctransformers
	
install-extra-mac:
	# brew install llvm libomp
	CXX=/usr/local/opt/llvm/bin/clang++ CC=/usr/local/opt/llvm/bin/clang pip install -U -r requirements_extra.txt
	pip show llama-cpp-python ctransformers
