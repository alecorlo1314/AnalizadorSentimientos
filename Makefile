install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black .

lint:
	pylint src/ entrenamiento.py --disable=R,C

train:
	python entrenamiento.py

eval:
	test -f ./Resultados/metricas.txt

	echo "## Metricas del Modelo" > reporte.md
	cat ./Resultados/metricas.txt >> reporte.md

	echo '\n## Matriz de Confusion' >> reporte.md
	echo '![Matriz de Confusion](./Resultados/confusion_matrix.png)' >> reporte.md

	echo '\n## Curva ROC' >> reporte.md
	echo '![Curva ROC](./Resultados/roc_curve.png)' >> reporte.md

	cml comment create reporte.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Actualizando los nuevos resultados"
	git push --force origin HEAD:update

configuracion_DVC_remoto:
	dvc remote add -f sentimiento_storage https://dagshub.com/alecorlo1234/AnalizadorSentimientos.dvc
	dvc remote default sentimiento_storage
	dvc remote modify sentimiento_storage auth basic
	dvc remote modify sentimiento_storage user alecorlo1234

hf-login:
	git fetch origin
	git checkout -B update
	git push -u origin update --force
	pip install -U "huggingface_hub[cli]"
	git config --global credential.helper store
	hf auth login --token $(HF) --add-to-git-credential

push-hub:
	hf upload upload alecorlo1234/AnalizadorSentimientos ./Aplicacion --repo-type=space --commit-message="Sincronizar Aplicacion"
	hf upload upload alecorlo1234/AnalizadorSentimientos ./Modelo /Modelo --repo-type=space --commit-message="Sincronizar Modelo"

deploy: hf-login push-hub