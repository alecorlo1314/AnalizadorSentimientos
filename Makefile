USER_NAME ?= github-actions[bot]
USER_EMAIL ?= github-actions[bot]@users.noreply.github.com

install:
	pip install -r requirements.txt

format:
	black --check src/ entrenamiento.py

lint:
	pylint src/ entrenamiento.py --fail-under=7

train:
	python entrenamiento.py

eval:
	cml comment create Resultados/reporte.md

update-branch:
	git config --global user.name "$(USER_NAME)"
	git config --global user.email "$(USER_EMAIL)"
	git checkout -B update
	git add Modelo/ Resultados/
	git commit -m "ci: actualizar modelo y resultados"
	git push --force origin update

configuracion_DVC_remoto:
	dvc remote add -f sentimientos_storage https://dagshub.com/alecorlo1234/AnalizadorSentimientos.dvc
	dvc remote default sentimientos_storage
	dvc remote modify sentimientos_storage auth basic
	dvc remote modify sentimientos_storage user alecorlo1234

hf-login:
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload alecorlo1234/AnalizadorSentimientos ./Aplicacion --repo-type=space --commit-message="Sincronizar Aplicacion"
	huggingface-cli upload alecorlo1234/AnalizadorSentimientos ./Modelo /Modelo --repo-type=space --commit-message="Sincronizar Modelo"

deploy: hf-login push-hub