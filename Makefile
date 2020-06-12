build:
	docker build -t retinanet -f Dockerfile .

run: build
	docker run \
		--gpus all \
		-it \
		--rm \
		--name retinanet \
		-v $(shell pwd):/workspace \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY=${DISPLAY} \
		-p 8888:8888 \
		-p 6006:6006 \
		retinanet
